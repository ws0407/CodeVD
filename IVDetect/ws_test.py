from utils.process import *
import pickle

def test_remove_comment():
    code = "static av_cold int vdadec_init(AVCodecContext *avctx)\n\n{\n\n    VDADecoderContext *ctx = avctx->priv_data;\n\n    struct vda_context *vda_ctx = &ctx->vda_ctx;\n\n    OSStatus status;\n\n    int ret;\n\n\n\n    ctx->h264_initialized = 0;\n\n\n\n    /* init pix_fmts of codec */\n\n    if (!ff_h264_vda_decoder.pix_fmts) {\n\n        if (kCFCoreFoundationVersionNumber < kCFCoreFoundationVersionNumber10_7)\n\n            ff_h264_vda_decoder.pix_fmts = vda_pixfmts_prior_10_7;\n\n        else\n\n            ff_h264_vda_decoder.pix_fmts = vda_pixfmts;\n\n    }\n\n\n\n    /* init vda */\n\n    memset(vda_ctx, 0, sizeof(struct vda_context));\n\n    vda_ctx->width = avctx->width;\n\n    vda_ctx->height = avctx->height;\n\n    vda_ctx->format = 'avc1';\n\n    vda_ctx->use_sync_decoding = 1;\n\n    vda_ctx->use_ref_buffer = 1;\n\n    ctx->pix_fmt = avctx->get_format(avctx, avctx->codec->pix_fmts);\n\n    switch (ctx->pix_fmt) {\n\n    case AV_PIX_FMT_UYVY422:\n\n        vda_ctx->cv_pix_fmt_type = '2vuy';\n\n        break;\n\n    case AV_PIX_FMT_YUYV422:\n\n        vda_ctx->cv_pix_fmt_type = 'yuvs';\n\n        break;\n\n    case AV_PIX_FMT_NV12:\n\n        vda_ctx->cv_pix_fmt_type = '420v';\n\n        break;\n\n    case AV_PIX_FMT_YUV420P:\n\n        vda_ctx->cv_pix_fmt_type = 'y420';\n\n        break;\n\n    default:\n\n        av_log(avctx, AV_LOG_ERROR, \"Unsupported pixel format: %d\\n\", avctx->pix_fmt);\n\n        goto failed;\n\n    }\n\n    status = ff_vda_create_decoder(vda_ctx,\n\n                                   avctx->extradata, avctx->extradata_size);\n\n    if (status != kVDADecoderNoErr) {\n\n        av_log(avctx, AV_LOG_ERROR,\n\n                \"Failed to init VDA decoder: %d.\\n\", status);\n\n        goto failed;\n\n    }\n\n    avctx->hwaccel_context = vda_ctx;\n\n\n\n    /* changes callback functions */\n\n    avctx->get_format = get_format;\n\n    avctx->get_buffer2 = get_buffer2;\n\n#if FF_API_GET_BUFFER\n\n    // force the old get_buffer to be empty\n\n    avctx->get_buffer = NULL;\n\n#endif\n\n\n\n    /* init H.264 decoder */\n\n    ret = ff_h264_decoder.init(avctx);\n\n    if (ret < 0) {\n\n        av_log(avctx, AV_LOG_ERROR, \"Failed to open H.264 decoder.\\n\");\n\n        goto failed;\n\n    }\n\n    ctx->h264_initialized = 1;\n\n\n\n    return 0;\n\n\n\nfailed:\n\n    vdadec_close(avctx);\n\n    return -1;\n\n}\n"
    print(code)
    out = remove_comment(code)
    print(out)


# test_remove_comment()

file = '/data/data/ws/NetworkTC/datasets/TrafficX-App_result_doc_flow/baidu.pkl'
with open(file, 'rb') as f:
    x = pickle.load(f)

print(len(x))
print(len(x))
print()