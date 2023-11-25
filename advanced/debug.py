from llama_index.callbacks import LlamaDebugHandler, CallbackManager

from config import DEBUG

if DEBUG:
    debug_handler = LlamaDebugHandler()
    cb_manager = CallbackManager([debug_handler])
else:
    debug_handler = None
    cb_manager = CallbackManager()
