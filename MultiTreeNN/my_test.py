import csv
import json
import os
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import re

from tqdm import tqdm

keywords = ['__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
            '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
            '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
            '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
            '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
            '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
            'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
            'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
            'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
            'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
            'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
            'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
            'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
            'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
            'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
            'wchar_t', 'while', 'xor', 'xor_eq', 'NULL',
            # others
            'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
            'BOOL', 'DWORD',
            # FFmpeg
            'av_always_inline', 'always_inline', 'av_cold', 'cold', 'av_extern_inline', 'av_warn_unused_result',
            'av_noinline', 'av_pure', 'av_const', 'av_flatten', 'av_unused', 'av_used', 'av_alias',
            'av_uninit', 'av_builtin_constant_p', 'av_builtin_constant_p', 'av_builtin_constant_p',
            'av_noreturn', 'CUDAAPI', 'attribute_align_arg',
            # qemu
            'coroutine_fn', 'always_inline', 'WINAPI', 'QEMU_WARN_UNUSED_RESULT', 'QEMU_NORETURN', 'CALLBACK']


def print_func(func: str = ""):
    func = func.replace('\n\n', '\n')
    func = re.sub(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)', '', func)
    func = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '""', func)
    func = re.sub(r'[\']([^\'\\\n]|\\.|\\\n)*[\']', '\'\'', func)
    print(func)


def count_dataset(filepath="/data/data/ws/CodeVD/MultiTreeNN/data/raw/dataset.json"):
    data = pd.read_json(filepath)
    print(data)
    print(data.columns)
    print(data["target"].value_counts())


def out_dataset_errors(
        input_json="/data/data/ws/CodeVD/IVDetect/data/FFMPeg_Qemu/function.json",
        key="func",
):
    """
    标准标识符：[800, 814, 1134, 13]
    err1: 函数无返回类型，joern解析没问题
    err2：代码括号不匹配，爬取代码不完整! target=0有43个, =1有757个, FFmpeg=355, qemu=455
    err3 & 4：需引入宏文件，加上宏文件FFmpeg和Qemu后可清除后两种解析bug
    """
    with open(input_json, 'r') as f:
        data = json.load(f)
    all_type = [0, 0, 0, 0]
    for idx, sample in tqdm(enumerate(data)):
        # if sample["project"] != "FFmpeg":
        #     continue
        func: str = sample[key]
        while '\n\n' in func:
            func = func.replace('\n\n', '\n')
        func = re.sub(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)', '', func)  # 去除/**/ //--注释
        func = re.sub(r'"([^"\\\n]|\\.|\\\n)*"', '""', func)                # 去除""包的字符串
        func = re.sub(r'\'([^\'\\\n]|\\.|\\\n)*\'', '\'\'', func)           # 去除''包的字符
        func_head = func[:func.find("(")]
        if func.count('(') != func.count(')') or func.count('{') != func.count('}'):
            print('{} brackets not match {}'.format('*' * 10, '*' * 10))
            print('[{}][target={}] {}'.format(sample['project'], sample['target'], func_head))
            all_type[0] += 1
            continue
        func_token = func_head.split()
        if len(func_token) <= 1:
            # print('{} len(func_token) <= 1 {}:'.format('*' * 10, '*' * 10))
            # print('[{}] {}'.format(sample['project'], func_head))
            all_type[1] += 1
            continue
        if len(func_token) == 4:
            if 'inline' in func_token or 'enum' in func_token or 'struct' in func_token or 'const' in func_token \
                    or '*' in func_token or 'void*' in func_token:
                continue
            if func_token[0] not in keywords or func_token[1] not in keywords or func_token[2] not in keywords:
                # print('{} len(func_token) == 4 {}:'.format('*' * 10, '*' * 10))
                # print('[{}] {}'.format(sample['project'], func_head))
                all_type[2] += 1
                continue
        if len(func_token) > 4:
            if 'inline' in func_token or 'enum' in func_token or 'struct' in func_token or 'const' in func_token \
                    or '*' in func_token:
                continue
            if func_token[0] not in keywords or func_token[1] not in keywords or \
                    func_token[2] not in keywords or func_token[3] not in keywords:
                # print('{} len(func_token) > 4 {}:'.format('*' * 10, '*' * 10))
                # print('[{}] {}'.format(sample['project'], func_head))
                all_type[3] += 1
                continue
        # func_name = func[:func.find("(")].split()[-1]
        # func_type = func.split()[0]
    print(len(data))
    print(all_type)


def get_no_func(
        input_json="/data/data/ws/CodeVD/IVDetect/data/FFMPeg_Qemu/function.json",
        key="func",
):
    """"""
    with open(input_json, 'r') as f:
        data = json.load(f)
    for idx, sample in tqdm(enumerate(data)):
        # if sample["project"] != "FFmpeg":
        #     continue
        func: str = sample[key]
        # while '\n\n' in func:
        #     func = func.replace('\n\n', '\n')
        func = re.sub(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)', '', func)  # 去除/**/ //--注释
        func = re.sub(r'"([^"\\\n]|\\.|\\\n)*"', '""', func)                # 去除""包的字符串
        func = re.sub(r'\'([^\'\\\n]|\\.|\\\n)*\'', '\'\'', func)           # 去除''包的字符
        func_head = func[:func.find("{")]
        func = func.replace('\n', '')
        func = func.replace('\r', '')
        func = func.replace('\t', '')
        func = func.replace(' ', '')
        left_bracket = func.find('(')
        right_bracket = func.find(')')
        left_brace = func.find('{')
        if left_bracket < 0 or right_bracket < 0 or left_brace < 0:
            print('{} brackets not find ({},{},{}) {}'.format(
                '*' * 10, left_bracket, right_bracket, left_brace, '*' * 10))
            print('[{}][target={}] {}'.format(sample['project'], sample['target'], func[:left_bracket]))
            continue
        if not (left_bracket < right_bracket < left_brace and func[left_brace-1] == ')'):
            print('{} brackets not match ({},{},{}) {}'.format(
                '*' * 10, left_bracket, right_bracket, left_brace, '*' * 10))
            print('[{}][target={}] {}'.format(sample['project'], sample['target'], func_head))


def test_joern_parse():
    from preprocessing import joern_parse
    cpg = joern_parse(
        joern_path='/data/data/ws/joern-1.0/joern-cli/',
        input_path='/data/data/ws/CodeVD/MultiTreeNN/test_data/multi/',
        output_path='/data/data/ws/CodeVD/MultiTreeNN/test_data/',
        file_name='multi'
    )
    print(cpg)


def test_joern_process():
    from preprocessing import joern_create
    out = joern_create(
        joern_path='/data/data/ws/joern-1.0/joern-cli/',
        script_path='/data/data/ws/CodeVD/MultiTreeNN/script/graph-for-funcs.sc',
        in_path='/data/data/ws/CodeVD/MultiTreeNN/data/cpg/',
        out_path='/data/data/ws/CodeVD/MultiTreeNN/data/cpg/out/',
        cpg_files=['{}_cpg.bin'.format(_) for _ in range(10)]
    )
    print(out)


def test_add_header():
    header_path = "/data/data/ws/CodeVD/MultiTreeNN/data/raw/"
    with open(header_path + 'FFmpeg.h', 'r') as f:
        FFmpeg_header = f.read()
    with open(header_path + 'qemu.h', 'r') as f:
        qemu_header = f.read()
    print(FFmpeg_header)
    print(qemu_header)


def test_json_process():
    file_path = "/data/data/ws/CodeVD/MultiTreeNN/data/cpg/0_cpg.pkl"
    with open(file_path, 'rb') as f:
        pkl = pickle.load(f)
    for func in pkl['cpg']:
        func = str(func)
        if '/*' in func:
            print(func)
    print(pkl)


func = "static void utf8_string(void)\n\n{\n\n    /*\n\n     * FIXME Current behavior for invalid UTF-8 sequences is\n\n     * incorrect.  This test expects current, incorrect results.\n\n     * They're all marked \"bug:\" below, and are to be replaced by\n\n     * correct ones as the bugs get fixed.\n\n     *\n\n     * The JSON parser rejects some invalid sequences, but accepts\n\n     * others without correcting the problem.\n\n     *\n\n     * We should either reject all invalid sequences, or minimize\n\n     * overlong sequences and replace all other invalid sequences by a\n\n     * suitable replacement character.  A common choice for\n\n     * replacement is U+FFFD.\n\n     *\n\n     * Problem: we can't easily deal with embedded U+0000.  Parsing\n\n     * the JSON string \"this \\\\u0000\" is fun\" yields \"this \\0 is fun\",\n\n     * which gets misinterpreted as NUL-terminated \"this \".  We should\n\n     * consider using overlong encoding \\xC0\\x80 for U+0000 (\"modified\n\n     * UTF-8\").\n\n     *\n\n     * Most test cases are scraped from Markus Kuhn's UTF-8 decoder\n\n     * capability and stress test at\n\n     * http://www.cl.cam.ac.uk/~mgk25/ucs/examples/UTF-8-test.txt\n\n     */\n\n    static const struct {\n\n        const char *json_in;\n\n        const char *utf8_out;\n\n        const char *json_out;   /* defaults to @json_in */\n\n        const char *utf8_in;    /* defaults to @utf8_out */\n\n    } test_cases[] = {\n\n        /*\n\n         * Bug markers used here:\n\n         * - bug: not corrected\n\n         *   JSON parser fails to correct invalid sequence(s)\n\n         * - bug: rejected\n\n         *   JSON parser rejects invalid sequence(s)\n\n         *   We may choose to define this as feature\n\n         * - bug: want \"...\"\n\n         *   JSON parser produces incorrect result, this is the\n\n         *   correct one, assuming replacement character U+FFFF\n\n         *   We may choose to reject instead of replace\n\n         */\n\n\n\n        /* 1  Some correct UTF-8 text */\n\n        {\n\n            /* a bit of German */\n\n            \"\\\"Falsches \\xC3\\x9C\" \"ben von Xylophonmusik qu\\xC3\\xA4lt\"\n\n            \" jeden gr\\xC3\\xB6\\xC3\\x9F\" \"eren Zwerg.\\\"\",\n\n            \"Falsches \\xC3\\x9C\" \"ben von Xylophonmusik qu\\xC3\\xA4lt\"\n\n            \" jeden gr\\xC3\\xB6\\xC3\\x9F\" \"eren Zwerg.\",\n\n            \"\\\"Falsches \\\\u00DCben von Xylophonmusik qu\\\\u00E4lt\"\n\n            \" jeden gr\\\\u00F6\\\\u00DFeren Zwerg.\\\"\",\n\n        },\n\n        {\n\n            /* a bit of Greek */\n\n            \"\\\"\\xCE\\xBA\\xE1\\xBD\\xB9\\xCF\\x83\\xCE\\xBC\\xCE\\xB5\\\"\",\n\n            \"\\xCE\\xBA\\xE1\\xBD\\xB9\\xCF\\x83\\xCE\\xBC\\xCE\\xB5\",\n\n            \"\\\"\\\\u03BA\\\\u1F79\\\\u03C3\\\\u03BC\\\\u03B5\\\"\",\n\n        },\n\n        /* 2  Boundary condition test cases */\n\n        /* 2.1  First possible sequence of a certain length */\n\n        /* 2.1.1  1 byte U+0000 */\n\n        {\n\n            \"\\\"\\\\u0000\\\"\",\n\n            \"\",                 /* bug: want overlong \"\\xC0\\x80\" */\n\n            \"\\\"\\\\u0000\\\"\",\n\n            \"\\xC0\\x80\",\n\n        },\n\n        /* 2.1.2  2 bytes U+0080 */\n\n        {\n\n            \"\\\"\\xC2\\x80\\\"\",\n\n            \"\\xC2\\x80\",\n\n            \"\\\"\\\\u0080\\\"\",\n\n        },\n\n        /* 2.1.3  3 bytes U+0800 */\n\n        {\n\n            \"\\\"\\xE0\\xA0\\x80\\\"\",\n\n            \"\\xE0\\xA0\\x80\",\n\n            \"\\\"\\\\u0800\\\"\",\n\n        },\n\n        /* 2.1.4  4 bytes U+10000 */\n\n        {\n\n            \"\\\"\\xF0\\x90\\x80\\x80\\\"\",\n\n            \"\\xF0\\x90\\x80\\x80\",\n\n            \"\\\"\\\\uD800\\\\uDC00\\\"\",\n\n        },\n\n        /* 2.1.5  5 bytes U+200000 */\n\n        {\n\n            \"\\\"\\xF8\\x88\\x80\\x80\\x80\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF8\\x88\\x80\\x80\\x80\",\n\n        },\n\n        /* 2.1.6  6 bytes U+4000000 */\n\n        {\n\n            \"\\\"\\xFC\\x84\\x80\\x80\\x80\\x80\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFC\\x84\\x80\\x80\\x80\\x80\",\n\n        },\n\n        /* 2.2  Last possible sequence of a certain length */\n\n        /* 2.2.1  1 byte U+007F */\n\n        {\n\n            \"\\\"\\x7F\\\"\",\n\n            \"\\x7F\",\n\n            \"\\\"\\\\u007F\\\"\",\n\n        },\n\n        /* 2.2.2  2 bytes U+07FF */\n\n        {\n\n            \"\\\"\\xDF\\xBF\\\"\",\n\n            \"\\xDF\\xBF\",\n\n            \"\\\"\\\\u07FF\\\"\",\n\n        },\n\n        /*\n\n         * 2.2.3  3 bytes U+FFFC\n\n         * The last possible sequence is actually U+FFFF.  But that's\n\n         * a noncharacter, and already covered by its own test case\n\n         * under 5.3.  Same for U+FFFE.  U+FFFD is the last character\n\n         * in the BMP, and covered under 2.3.  Because of U+FFFD's\n\n         * special role as replacement character, it's worth testing\n\n         * U+FFFC here.\n\n         */\n\n        {\n\n            \"\\\"\\xEF\\xBF\\xBC\\\"\",\n\n            \"\\xEF\\xBF\\xBC\",\n\n            \"\\\"\\\\uFFFC\\\"\",\n\n        },\n\n        /* 2.2.4  4 bytes U+1FFFFF */\n\n        {\n\n            \"\\\"\\xF7\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF7\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 2.2.5  5 bytes U+3FFFFFF */\n\n        {\n\n            \"\\\"\\xFB\\xBF\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFB\\xBF\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 2.2.6  6 bytes U+7FFFFFFF */\n\n        {\n\n            \"\\\"\\xFD\\xBF\\xBF\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFD\\xBF\\xBF\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 2.3  Other boundary conditions */\n\n        {\n\n            /* last one before surrogate range: U+D7FF */\n\n            \"\\\"\\xED\\x9F\\xBF\\\"\",\n\n            \"\\xED\\x9F\\xBF\",\n\n            \"\\\"\\\\uD7FF\\\"\",\n\n        },\n\n        {\n\n            /* first one after surrogate range: U+E000 */\n\n            \"\\\"\\xEE\\x80\\x80\\\"\",\n\n            \"\\xEE\\x80\\x80\",\n\n            \"\\\"\\\\uE000\\\"\",\n\n        },\n\n        {\n\n            /* last one in BMP: U+FFFD */\n\n            \"\\\"\\xEF\\xBF\\xBD\\\"\",\n\n            \"\\xEF\\xBF\\xBD\",\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* last one in last plane: U+10FFFD */\n\n            \"\\\"\\xF4\\x8F\\xBF\\xBD\\\"\",\n\n            \"\\xF4\\x8F\\xBF\\xBD\",\n\n            \"\\\"\\\\uDBFF\\\\uDFFD\\\"\"\n\n        },\n\n        {\n\n            /* first one beyond Unicode range: U+110000 */\n\n            \"\\\"\\xF4\\x90\\x80\\x80\\\"\",\n\n            \"\\xF4\\x90\\x80\\x80\",\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3  Malformed sequences */\n\n        /* 3.1  Unexpected continuation bytes */\n\n        /* 3.1.1  First continuation byte */\n\n        {\n\n            \"\\\"\\x80\\\"\",\n\n            \"\\x80\",             /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.2  Last continuation byte */\n\n        {\n\n            \"\\\"\\xBF\\\"\",\n\n            \"\\xBF\",             /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.3  2 continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\xBF\\\"\",\n\n            \"\\x80\\xBF\",         /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.4  3 continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\xBF\\x80\\\"\",\n\n            \"\\x80\\xBF\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.5  4 continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\xBF\\x80\\xBF\\\"\",\n\n            \"\\x80\\xBF\\x80\\xBF\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.6  5 continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\xBF\\x80\\xBF\\x80\\\"\",\n\n            \"\\x80\\xBF\\x80\\xBF\\x80\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.7  6 continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\xBF\\x80\\xBF\\x80\\xBF\\\"\",\n\n            \"\\x80\\xBF\\x80\\xBF\\x80\\xBF\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.8  7 continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\xBF\\x80\\xBF\\x80\\xBF\\x80\\\"\",\n\n            \"\\x80\\xBF\\x80\\xBF\\x80\\xBF\\x80\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.1.9  Sequence of all 64 possible continuation bytes */\n\n        {\n\n            \"\\\"\\x80\\x81\\x82\\x83\\x84\\x85\\x86\\x87\"\n\n            \"\\x88\\x89\\x8A\\x8B\\x8C\\x8D\\x8E\\x8F\"\n\n            \"\\x90\\x91\\x92\\x93\\x94\\x95\\x96\\x97\"\n\n            \"\\x98\\x99\\x9A\\x9B\\x9C\\x9D\\x9E\\x9F\"\n\n            \"\\xA0\\xA1\\xA2\\xA3\\xA4\\xA5\\xA6\\xA7\"\n\n            \"\\xA8\\xA9\\xAA\\xAB\\xAC\\xAD\\xAE\\xAF\"\n\n            \"\\xB0\\xB1\\xB2\\xB3\\xB4\\xB5\\xB6\\xB7\"\n\n            \"\\xB8\\xB9\\xBA\\xBB\\xBC\\xBD\\xBE\\xBF\\\"\",\n\n             /* bug: not corrected */\n\n            \"\\x80\\x81\\x82\\x83\\x84\\x85\\x86\\x87\"\n\n            \"\\x88\\x89\\x8A\\x8B\\x8C\\x8D\\x8E\\x8F\"\n\n            \"\\x90\\x91\\x92\\x93\\x94\\x95\\x96\\x97\"\n\n            \"\\x98\\x99\\x9A\\x9B\\x9C\\x9D\\x9E\\x9F\"\n\n            \"\\xA0\\xA1\\xA2\\xA3\\xA4\\xA5\\xA6\\xA7\"\n\n            \"\\xA8\\xA9\\xAA\\xAB\\xAC\\xAD\\xAE\\xAF\"\n\n            \"\\xB0\\xB1\\xB2\\xB3\\xB4\\xB5\\xB6\\xB7\"\n\n            \"\\xB8\\xB9\\xBA\\xBB\\xBC\\xBD\\xBE\\xBF\",\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\"\n\n        },\n\n        /* 3.2  Lonely start characters */\n\n        /* 3.2.1  All 32 first bytes of 2-byte sequences, followed by space */\n\n        {\n\n            \"\\\"\\xC0 \\xC1 \\xC2 \\xC3 \\xC4 \\xC5 \\xC6 \\xC7 \"\n\n            \"\\xC8 \\xC9 \\xCA \\xCB \\xCC \\xCD \\xCE \\xCF \"\n\n            \"\\xD0 \\xD1 \\xD2 \\xD3 \\xD4 \\xD5 \\xD6 \\xD7 \"\n\n            \"\\xD8 \\xD9 \\xDA \\xDB \\xDC \\xDD \\xDE \\xDF \\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \"\n\n            \"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \"\n\n            \"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \"\n\n            \"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\"\",\n\n            \"\\xC0 \\xC1 \\xC2 \\xC3 \\xC4 \\xC5 \\xC6 \\xC7 \"\n\n            \"\\xC8 \\xC9 \\xCA \\xCB \\xCC \\xCD \\xCE \\xCF \"\n\n            \"\\xD0 \\xD1 \\xD2 \\xD3 \\xD4 \\xD5 \\xD6 \\xD7 \"\n\n            \"\\xD8 \\xD9 \\xDA \\xDB \\xDC \\xDD \\xDE \\xDF \",\n\n        },\n\n        /* 3.2.2  All 16 first bytes of 3-byte sequences, followed by space */\n\n        {\n\n            \"\\\"\\xE0 \\xE1 \\xE2 \\xE3 \\xE4 \\xE5 \\xE6 \\xE7 \"\n\n            \"\\xE8 \\xE9 \\xEA \\xEB \\xEC \\xED \\xEE \\xEF \\\"\",\n\n            /* bug: not corrected */\n\n            \"\\xE0 \\xE1 \\xE2 \\xE3 \\xE4 \\xE5 \\xE6 \\xE7 \"\n\n            \"\\xE8 \\xE9 \\xEA \\xEB \\xEC \\xED \\xEE \\xEF \",\n\n            \"\\\"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \"\n\n            \"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\"\",\n\n        },\n\n        /* 3.2.3  All 8 first bytes of 4-byte sequences, followed by space */\n\n        {\n\n            \"\\\"\\xF0 \\xF1 \\xF2 \\xF3 \\xF4 \\xF5 \\xF6 \\xF7 \\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\"\",\n\n            \"\\xF0 \\xF1 \\xF2 \\xF3 \\xF4 \\xF5 \\xF6 \\xF7 \",\n\n        },\n\n        /* 3.2.4  All 4 first bytes of 5-byte sequences, followed by space */\n\n        {\n\n            \"\\\"\\xF8 \\xF9 \\xFA \\xFB \\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD \\\\uFFFD \\\\uFFFD \\\\uFFFD \\\"\",\n\n            \"\\xF8 \\xF9 \\xFA \\xFB \",\n\n        },\n\n        /* 3.2.5  All 2 first bytes of 6-byte sequences, followed by space */\n\n        {\n\n            \"\\\"\\xFC \\xFD \\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD \\\\uFFFD \\\"\",\n\n            \"\\xFC \\xFD \",\n\n        },\n\n        /* 3.3  Sequences with last continuation byte missing */\n\n        /* 3.3.1  2-byte sequence with last byte missing (U+0000) */\n\n        {\n\n            \"\\\"\\xC0\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xC0\",\n\n        },\n\n        /* 3.3.2  3-byte sequence with last byte missing (U+0000) */\n\n        {\n\n            \"\\\"\\xE0\\x80\\\"\",\n\n            \"\\xE0\\x80\",           /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.3.3  4-byte sequence with last byte missing (U+0000) */\n\n        {\n\n            \"\\\"\\xF0\\x80\\x80\\\"\",\n\n            \"\\xF0\\x80\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.3.4  5-byte sequence with last byte missing (U+0000) */\n\n        {\n\n            \"\\\"\\xF8\\x80\\x80\\x80\\\"\",\n\n            NULL,                   /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF8\\x80\\x80\\x80\",\n\n        },\n\n        /* 3.3.5  6-byte sequence with last byte missing (U+0000) */\n\n        {\n\n            \"\\\"\\xFC\\x80\\x80\\x80\\x80\\\"\",\n\n            NULL,                        /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFC\\x80\\x80\\x80\\x80\",\n\n        },\n\n        /* 3.3.6  2-byte sequence with last byte missing (U+07FF) */\n\n        {\n\n            \"\\\"\\xDF\\\"\",\n\n            \"\\xDF\",             /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.3.7  3-byte sequence with last byte missing (U+FFFF) */\n\n        {\n\n            \"\\\"\\xEF\\xBF\\\"\",\n\n            \"\\xEF\\xBF\",           /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 3.3.8  4-byte sequence with last byte missing (U+1FFFFF) */\n\n        {\n\n            \"\\\"\\xF7\\xBF\\xBF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF7\\xBF\\xBF\",\n\n        },\n\n        /* 3.3.9  5-byte sequence with last byte missing (U+3FFFFFF) */\n\n        {\n\n            \"\\\"\\xFB\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,                 /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFB\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 3.3.10  6-byte sequence with last byte missing (U+7FFFFFFF) */\n\n        {\n\n            \"\\\"\\xFD\\xBF\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,                        /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFD\\xBF\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 3.4  Concatenation of incomplete sequences */\n\n        {\n\n            \"\\\"\\xC0\\xE0\\x80\\xF0\\x80\\x80\\xF8\\x80\\x80\\x80\\xFC\\x80\\x80\\x80\\x80\"\n\n            \"\\xDF\\xEF\\xBF\\xF7\\xBF\\xBF\\xFB\\xBF\\xBF\\xBF\\xFD\\xBF\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n            \"\\xC0\\xE0\\x80\\xF0\\x80\\x80\\xF8\\x80\\x80\\x80\\xFC\\x80\\x80\\x80\\x80\"\n\n            \"\\xDF\\xEF\\xBF\\xF7\\xBF\\xBF\\xFB\\xBF\\xBF\\xBF\\xFD\\xBF\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 3.5  Impossible bytes */\n\n        {\n\n            \"\\\"\\xFE\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFE\",\n\n        },\n\n        {\n\n            \"\\\"\\xFF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFF\",\n\n        },\n\n        {\n\n            \"\\\"\\xFE\\xFE\\xFF\\xFF\\\"\",\n\n            NULL,                 /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n            \"\\xFE\\xFE\\xFF\\xFF\",\n\n        },\n\n        /* 4  Overlong sequences */\n\n        /* 4.1  Overlong '/' */\n\n        {\n\n            \"\\\"\\xC0\\xAF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xC0\\xAF\",\n\n        },\n\n        {\n\n            \"\\\"\\xE0\\x80\\xAF\\\"\",\n\n            \"\\xE0\\x80\\xAF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            \"\\\"\\xF0\\x80\\x80\\xAF\\\"\",\n\n            \"\\xF0\\x80\\x80\\xAF\",  /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            \"\\\"\\xF8\\x80\\x80\\x80\\xAF\\\"\",\n\n            NULL,                        /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF8\\x80\\x80\\x80\\xAF\",\n\n        },\n\n        {\n\n            \"\\\"\\xFC\\x80\\x80\\x80\\x80\\xAF\\\"\",\n\n            NULL,                               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFC\\x80\\x80\\x80\\x80\\xAF\",\n\n        },\n\n        /*\n\n         * 4.2  Maximum overlong sequences\n\n         * Highest Unicode value that is still resulting in an\n\n         * overlong sequence if represented with the given number of\n\n         * bytes.  This is a boundary test for safe UTF-8 decoders.\n\n         */\n\n        {\n\n            /* \\U+007F */\n\n            \"\\\"\\xC1\\xBF\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xC1\\xBF\",\n\n        },\n\n        {\n\n            /* \\U+07FF */\n\n            \"\\\"\\xE0\\x9F\\xBF\\\"\",\n\n            \"\\xE0\\x9F\\xBF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /*\n\n             * \\U+FFFC\n\n             * The actual maximum would be U+FFFF, but that's a\n\n             * noncharacter.  Testing U+FFFC seems more useful.  See\n\n             * also 2.2.3\n\n             */\n\n            \"\\\"\\xF0\\x8F\\xBF\\xBC\\\"\",\n\n            \"\\xF0\\x8F\\xBF\\xBC\",   /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+1FFFFF */\n\n            \"\\\"\\xF8\\x87\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,                        /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF8\\x87\\xBF\\xBF\\xBF\",\n\n        },\n\n        {\n\n            /* \\U+3FFFFFF */\n\n            \"\\\"\\xFC\\x83\\xBF\\xBF\\xBF\\xBF\\\"\",\n\n            NULL,                               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFC\\x83\\xBF\\xBF\\xBF\\xBF\",\n\n        },\n\n        /* 4.3  Overlong representation of the NUL character */\n\n        {\n\n            /* \\U+0000 */\n\n            \"\\\"\\xC0\\x80\\\"\",\n\n            NULL,               /* bug: rejected */\n\n            \"\\\"\\\\u0000\\\"\",\n\n            \"\\xC0\\x80\",\n\n        },\n\n        {\n\n            /* \\U+0000 */\n\n            \"\\\"\\xE0\\x80\\x80\\\"\",\n\n            \"\\xE0\\x80\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+0000 */\n\n            \"\\\"\\xF0\\x80\\x80\\x80\\\"\",\n\n            \"\\xF0\\x80\\x80\\x80\",   /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+0000 */\n\n            \"\\\"\\xF8\\x80\\x80\\x80\\x80\\\"\",\n\n            NULL,                        /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xF8\\x80\\x80\\x80\\x80\",\n\n        },\n\n        {\n\n            /* \\U+0000 */\n\n            \"\\\"\\xFC\\x80\\x80\\x80\\x80\\x80\\\"\",\n\n            NULL,                               /* bug: rejected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n            \"\\xFC\\x80\\x80\\x80\\x80\\x80\",\n\n        },\n\n        /* 5  Illegal code positions */\n\n        /* 5.1  Single UTF-16 surrogates */\n\n        {\n\n            /* \\U+D800 */\n\n            \"\\\"\\xED\\xA0\\x80\\\"\",\n\n            \"\\xED\\xA0\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DB7F */\n\n            \"\\\"\\xED\\xAD\\xBF\\\"\",\n\n            \"\\xED\\xAD\\xBF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DB80 */\n\n            \"\\\"\\xED\\xAE\\x80\\\"\",\n\n            \"\\xED\\xAE\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DBFF */\n\n            \"\\\"\\xED\\xAF\\xBF\\\"\",\n\n            \"\\xED\\xAF\\xBF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DC00 */\n\n            \"\\\"\\xED\\xB0\\x80\\\"\",\n\n            \"\\xED\\xB0\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DF80 */\n\n            \"\\\"\\xED\\xBE\\x80\\\"\",\n\n            \"\\xED\\xBE\\x80\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DFFF */\n\n            \"\\\"\\xED\\xBF\\xBF\\\"\",\n\n            \"\\xED\\xBF\\xBF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* 5.2  Paired UTF-16 surrogates */\n\n        {\n\n            /* \\U+D800\\U+DC00 */\n\n            \"\\\"\\xED\\xA0\\x80\\xED\\xB0\\x80\\\"\",\n\n            \"\\xED\\xA0\\x80\\xED\\xB0\\x80\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+D800\\U+DFFF */\n\n            \"\\\"\\xED\\xA0\\x80\\xED\\xBF\\xBF\\\"\",\n\n            \"\\xED\\xA0\\x80\\xED\\xBF\\xBF\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DB7F\\U+DC00 */\n\n            \"\\\"\\xED\\xAD\\xBF\\xED\\xB0\\x80\\\"\",\n\n            \"\\xED\\xAD\\xBF\\xED\\xB0\\x80\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DB7F\\U+DFFF */\n\n            \"\\\"\\xED\\xAD\\xBF\\xED\\xBF\\xBF\\\"\",\n\n            \"\\xED\\xAD\\xBF\\xED\\xBF\\xBF\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DB80\\U+DC00 */\n\n            \"\\\"\\xED\\xAE\\x80\\xED\\xB0\\x80\\\"\",\n\n            \"\\xED\\xAE\\x80\\xED\\xB0\\x80\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DB80\\U+DFFF */\n\n            \"\\\"\\xED\\xAE\\x80\\xED\\xBF\\xBF\\\"\",\n\n            \"\\xED\\xAE\\x80\\xED\\xBF\\xBF\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DBFF\\U+DC00 */\n\n            \"\\\"\\xED\\xAF\\xBF\\xED\\xB0\\x80\\\"\",\n\n            \"\\xED\\xAF\\xBF\\xED\\xB0\\x80\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+DBFF\\U+DFFF */\n\n            \"\\\"\\xED\\xAF\\xBF\\xED\\xBF\\xBF\\\"\",\n\n            \"\\xED\\xAF\\xBF\\xED\\xBF\\xBF\", /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        /* 5.3  Other illegal code positions */\n\n        /* BMP noncharacters */\n\n        {\n\n            /* \\U+FFFE */\n\n            \"\\\"\\xEF\\xBF\\xBE\\\"\",\n\n            \"\\xEF\\xBF\\xBE\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* \\U+FFFF */\n\n            \"\\\"\\xEF\\xBF\\xBF\\\"\",\n\n            \"\\xEF\\xBF\\xBF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* U+FDD0 */\n\n            \"\\\"\\xEF\\xB7\\x90\\\"\",\n\n            \"\\xEF\\xB7\\x90\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        {\n\n            /* U+FDEF */\n\n            \"\\\"\\xEF\\xB7\\xAF\\\"\",\n\n            \"\\xEF\\xB7\\xAF\",     /* bug: not corrected */\n\n            \"\\\"\\\\uFFFD\\\"\",\n\n        },\n\n        /* Plane 1 .. 16 noncharacters */\n\n        {\n\n            /* U+1FFFE U+1FFFF U+2FFFE U+2FFFF ... U+10FFFE U+10FFFF */\n\n            \"\\\"\\xF0\\x9F\\xBF\\xBE\\xF0\\x9F\\xBF\\xBF\"\n\n            \"\\xF0\\xAF\\xBF\\xBE\\xF0\\xAF\\xBF\\xBF\"\n\n            \"\\xF0\\xBF\\xBF\\xBE\\xF0\\xBF\\xBF\\xBF\"\n\n            \"\\xF1\\x8F\\xBF\\xBE\\xF1\\x8F\\xBF\\xBF\"\n\n            \"\\xF1\\x9F\\xBF\\xBE\\xF1\\x9F\\xBF\\xBF\"\n\n            \"\\xF1\\xAF\\xBF\\xBE\\xF1\\xAF\\xBF\\xBF\"\n\n            \"\\xF1\\xBF\\xBF\\xBE\\xF1\\xBF\\xBF\\xBF\"\n\n            \"\\xF2\\x8F\\xBF\\xBE\\xF2\\x8F\\xBF\\xBF\"\n\n            \"\\xF2\\x9F\\xBF\\xBE\\xF2\\x9F\\xBF\\xBF\"\n\n            \"\\xF2\\xAF\\xBF\\xBE\\xF2\\xAF\\xBF\\xBF\"\n\n            \"\\xF2\\xBF\\xBF\\xBE\\xF2\\xBF\\xBF\\xBF\"\n\n            \"\\xF3\\x8F\\xBF\\xBE\\xF3\\x8F\\xBF\\xBF\"\n\n            \"\\xF3\\x9F\\xBF\\xBE\\xF3\\x9F\\xBF\\xBF\"\n\n            \"\\xF3\\xAF\\xBF\\xBE\\xF3\\xAF\\xBF\\xBF\"\n\n            \"\\xF3\\xBF\\xBF\\xBE\\xF3\\xBF\\xBF\\xBF\"\n\n            \"\\xF4\\x8F\\xBF\\xBE\\xF4\\x8F\\xBF\\xBF\\\"\",\n\n            /* bug: not corrected */\n\n            \"\\xF0\\x9F\\xBF\\xBE\\xF0\\x9F\\xBF\\xBF\"\n\n            \"\\xF0\\xAF\\xBF\\xBE\\xF0\\xAF\\xBF\\xBF\"\n\n            \"\\xF0\\xBF\\xBF\\xBE\\xF0\\xBF\\xBF\\xBF\"\n\n            \"\\xF1\\x8F\\xBF\\xBE\\xF1\\x8F\\xBF\\xBF\"\n\n            \"\\xF1\\x9F\\xBF\\xBE\\xF1\\x9F\\xBF\\xBF\"\n\n            \"\\xF1\\xAF\\xBF\\xBE\\xF1\\xAF\\xBF\\xBF\"\n\n            \"\\xF1\\xBF\\xBF\\xBE\\xF1\\xBF\\xBF\\xBF\"\n\n            \"\\xF2\\x8F\\xBF\\xBE\\xF2\\x8F\\xBF\\xBF\"\n\n            \"\\xF2\\x9F\\xBF\\xBE\\xF2\\x9F\\xBF\\xBF\"\n\n            \"\\xF2\\xAF\\xBF\\xBE\\xF2\\xAF\\xBF\\xBF\"\n\n            \"\\xF2\\xBF\\xBF\\xBE\\xF2\\xBF\\xBF\\xBF\"\n\n            \"\\xF3\\x8F\\xBF\\xBE\\xF3\\x8F\\xBF\\xBF\"\n\n            \"\\xF3\\x9F\\xBF\\xBE\\xF3\\x9F\\xBF\\xBF\"\n\n            \"\\xF3\\xAF\\xBF\\xBE\\xF3\\xAF\\xBF\\xBF\"\n\n            \"\\xF3\\xBF\\xBF\\xBE\\xF3\\xBF\\xBF\\xBF\"\n\n            \"\\xF4\\x8F\\xBF\\xBE\\xF4\\x8F\\xBF\\xBF\",\n\n            \"\\\"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\"\n\n            \"\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\\uFFFD\\\"\",\n\n        },\n\n        {}\n\n    };\n\n    int i;\n\n    QObject *obj;\n\n    QString *str;\n\n    const char *json_in, *utf8_out, *utf8_in, *json_out;\n\n\n\n    for (i = 0; test_cases[i].json_in; i++) {\n\n        json_in = test_cases[i].json_in;\n\n        utf8_out = test_cases[i].utf8_out;\n\n        utf8_in = test_cases[i].utf8_in ?: test_cases[i].utf8_out;\n\n        json_out = test_cases[i].json_out ?: test_cases[i].json_in;\n\n\n\n        obj = qobject_from_json(json_in, NULL);\n\n        if (utf8_out) {\n\n            str = qobject_to_qstring(obj);\n\n            g_assert(str);\n\n            g_assert_cmpstr(qstring_get_str(str), ==, utf8_out);\n\n        } else {\n\n            g_assert(!obj);\n\n        }\n\n        qobject_decref(obj);\n\n\n\n        obj = QOBJECT(qstring_from_str(utf8_in));\n\n        str = qobject_to_json(obj);\n\n        if (json_out) {\n\n            g_assert(str);\n\n            g_assert_cmpstr(qstring_get_str(str), ==, json_out);\n\n        } else {\n\n            g_assert(!str);\n\n        }\n\n        QDECREF(str);\n\n        qobject_decref(obj);\n\n\n\n        /*\n\n         * Disabled, because qobject_from_json() is buggy, and I can't\n\n         * be bothered to add the expected incorrect results.\n\n         * FIXME Enable once these bugs have been fixed.\n\n         */\n\n        if (0 && json_out != json_in) {\n\n            obj = qobject_from_json(json_out, NULL);\n\n            str = qobject_to_qstring(obj);\n\n            g_assert(str);\n\n            g_assert_cmpstr(qstring_get_str(str), ==, utf8_out);\n\n        }\n\n    }\n\n}\n"
print(func)
print("\n\n*************************************************************\n\n")
func = re.sub(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)', '', func)  # 去除/**/ //--注释
func = re.sub(r'"([^"\\\n]|\\.|\\\n)*"', '""', func)  # 去除""包的字符串
func = re.sub(r'\n+', '\n', func)
print(func)

# test_json_process()
# test_add_header()
# test_joern_process()
# test_joern_parse()
# get_no_func()

# out_dataset_errors()
#
# count_dataset()
#
#
# def format_id(string):
#     string = re.sub(r'io\.shiftleft\.codepropertygraph\.generated\.', '', string)
#     string = re.sub(r'(\[label)(\D+)(\d+)\]', lambda match: f'@{match.group(3)}', string)
#     # string = re.sub(r'\d+', lambda match: f'@{match.group()}', string)
#     return string
#
#
# # 测试示例
# print(format_id("io.shiftleft.codepropertygraph.generated.edges.Ast@2b423"))
# print(format_id("\"io.shiftleft.codepropertygraph.generated.edges.Ast@2b062\","))
# s = """"function" : "ff_af_queue_init",
#       "id" : "io.shiftleft.codepropertygraph.generated.nodes.Method[label=METHOD; id=2305843009213694033]",
#       "AST" : [
#         {
#           "id" : "io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn[label=METHOD_PARAMETER_IN; id=2305843009213694034]",
#           "edges" : [
#             {
#               "id" : "io.shiftleft.codepropertygraph.generated.edges.Ast@2b062",
#               "in" : "io.shiftleft.codepropertygraph.generated.nodes.MethodParameterIn[label=METHOD_PARAMETER_IN; id=2305843009213694034]",
#               "out" : "io.shiftleft.codepropertygraph.generated.nodes.Method[label=METHOD; id=2305843009213694033]"
#             }
#           ],
#           "properties" : [
#             {
#               "key" : "ORDER",
#               "value" : "1"
#             },
#             {
#               "key" : "CODE",
#               "value" : "AVCodecContext *avctx"
#             }
#           ]
#         },"""
# print(s)
# print(format_id(s))

# from IVDetect.main import MyDatset, collate_batch
#
# test_path = '/data/data/ws/CodeVD/IVDetect/data/Fan_et_al/test_data/'
# test_files = [f for f in os.listdir(test_path) if
#               os.path.isfile(os.path.join(test_path, f))]
# test_dataset = MyDatset(test_files, test_path)
#
# test_loader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)
#
# print('len:', len(test_loader))

# import pickle
# import json
#
# with open('data/cpg/0_cpg.pkl', 'rb') as f:
#     data: pd.DataFrame = pickle.load(f)
#
# print(data.columns)
# print(data.head(2))
#
# with open('data/cpg/test.json', 'w', encoding="utf-8") as f:
#     json.dump(data.head(10).to_json(), f, ensure_ascii=False)


# print(cpg['cpg'].iloc[6])
# print(len(cpg))


# import pickle
#
# from utils.functions import parse_to_nodes
#
# with open('/data/data/ws/CodeVD/MultiTreeNN/data/cpg/0_cpg.pkl', 'rb') as f:
#     data: pd.DataFrame = pickle.load(f)
# # data["nodes"] = data.apply(lambda row: parse_to_nodes(row.cpg, 205), axis=1)
# # print()
# for funcs in data["cpg"]:
#     for k, v in funcs.items():
#         # print(len(v))
#         if 'ASN1_TYPE_set' in str(v[0]):
#             print(k, v[0])
#             break
