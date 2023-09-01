import torch

from utils.process import *
import os
from torch.utils.data import DataLoader
from main import MyDatset, collate_batch

def test_remove_comment():
    # code = "static av_cold int vdadec_init(AVCodecContext *avctx)\n\n{\n\n    VDADecoderContext *ctx = avctx->priv_data;\n\n    struct vda_context *vda_ctx = &ctx->vda_ctx;\n\n    OSStatus status;\n\n    int ret;\n\n\n\n    ctx->h264_initialized = 0;\n\n\n\n    /* init pix_fmts of codec */\n\n    if (!ff_h264_vda_decoder.pix_fmts) {\n\n        if (kCFCoreFoundationVersionNumber < kCFCoreFoundationVersionNumber10_7)\n\n            ff_h264_vda_decoder.pix_fmts = vda_pixfmts_prior_10_7;\n\n        else\n\n            ff_h264_vda_decoder.pix_fmts = vda_pixfmts;\n\n    }\n\n\n\n    /* init vda */\n\n    memset(vda_ctx, 0, sizeof(struct vda_context));\n\n    vda_ctx->width = avctx->width;\n\n    vda_ctx->height = avctx->height;\n\n    vda_ctx->format = 'avc1';\n\n    vda_ctx->use_sync_decoding = 1;\n\n    vda_ctx->use_ref_buffer = 1;\n\n    ctx->pix_fmt = avctx->get_format(avctx, avctx->codec->pix_fmts);\n\n    switch (ctx->pix_fmt) {\n\n    case AV_PIX_FMT_UYVY422:\n\n        vda_ctx->cv_pix_fmt_type = '2vuy';\n\n        break;\n\n    case AV_PIX_FMT_YUYV422:\n\n        vda_ctx->cv_pix_fmt_type = 'yuvs';\n\n        break;\n\n    case AV_PIX_FMT_NV12:\n\n        vda_ctx->cv_pix_fmt_type = '420v';\n\n        break;\n\n    case AV_PIX_FMT_YUV420P:\n\n        vda_ctx->cv_pix_fmt_type = 'y420';\n\n        break;\n\n    default:\n\n        av_log(avctx, AV_LOG_ERROR, \"Unsupported pixel format: %d\\n\", avctx->pix_fmt);\n\n        goto failed;\n\n    }\n\n    status = ff_vda_create_decoder(vda_ctx,\n\n                                   avctx->extradata, avctx->extradata_size);\n\n    if (status != kVDADecoderNoErr) {\n\n        av_log(avctx, AV_LOG_ERROR,\n\n                \"Failed to init VDA decoder: %d.\\n\", status);\n\n        goto failed;\n\n    }\n\n    avctx->hwaccel_context = vda_ctx;\n\n\n\n    /* changes callback functions */\n\n    avctx->get_format = get_format;\n\n    avctx->get_buffer2 = get_buffer2;\n\n#if FF_API_GET_BUFFER\n\n    // force the old get_buffer to be empty\n\n    avctx->get_buffer = NULL;\n\n#endif\n\n\n\n    /* init H.264 decoder */\n\n    ret = ff_h264_decoder.init(avctx);\n\n    if (ret < 0) {\n\n        av_log(avctx, AV_LOG_ERROR, \"Failed to open H.264 decoder.\\n\");\n\n        goto failed;\n\n    }\n\n    ctx->h264_initialized = 1;\n\n\n\n    return 0;\n\n\n\nfailed:\n\n    vdadec_close(avctx);\n\n    return -1;\n\n}\n"

    code = "static void v4l2_free_buffer(void *opaque, uint8_t *unused)\n\n{\n\n    V4L2Buffer* avbuf = opaque;\n\n    V4L2m2mContext *s = buf_to_m2mctx(avbuf);\n\n\n\n    if (atomic_fetch_sub(&avbuf->context_refcount, 1) == 1) {\n\n        atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel);\n\n\n\n        if (s->reinit) {\n\n            if (!atomic_load(&s->refcount))\n\n                sem_post(&s->refsync);\n\n        } else if (avbuf->context->streamon)\n\n            ff_v4l2_buffer_enqueue(avbuf);\n\n\n\n        av_buffer_unref(&avbuf->context_ref);\n\n    }\n\n}\n"

    code = code.replace('\n\n', '\n')
    print(code)

    code = code.splitlines()
    print(code)
    code = remove_comment(code)
    print('\n\n*****************************************\n\n')
    print(code)
    code = " ".join(code)
    code = re.sub('[^a-zA-Z0-9]', ' ', code)
    print(code)
    code = re.sub(' +', ' ', code)
    print(code)
    code = re.sub(r"([A-Z])", r" \1", code).split()
    print(code)

    code = merge_code(code)
    print(code)


def test_load_dataset():
    test_path = 'data/Fan_et_al/test_data/'
    test_files = [f for f in os.listdir(test_path) if
                  os.path.isfile(os.path.join(test_path, f))]
    test_dataset = MyDatset(test_files, test_path)

    test_loader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)

    print('len:', len(test_loader))

    for index, graph in enumerate(tqdm(test_loader)):
        print(index)
        print(graph.shape)


# test_load_dataset()

x = torch.load('/data/data/ws/CodeVD/IVDetect/data/Fan_et_al/test_data/data_0.pt')

print(1)
print(x.shape)
