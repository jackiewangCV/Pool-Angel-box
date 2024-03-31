import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading

class TRTInference:
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, "rb") as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs  = []
        cuda_inputs  = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        input_shapes = []
        out_shapes = []
        
        for binding in engine:
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            binding_shape = engine.get_tensor_shape(binding)
            print(f"Name binding: {binding}, shape: {binding_shape}, size: {size}")

            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                input_shapes.append(binding_shape)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                out_shapes.append(binding_shape)

        # store
        self.stream  = stream
        self.context = context
        self.engine  = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.input_shapes = input_shapes
        self.out_shapes = out_shapes
        self.batch_size = batch_size

    def infer(self, input_data):
        threading.Thread.__init__(self)
        self.cfx.push()

        # restore
        stream  = self.stream
        context = self.context
        engine  = self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # # -----read image
        # np.copyto(host_inputs[0], input_data.ravel())
        # # -----inference
        # cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # # context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # stream.synchronize()
        # # -----get output
        # trt_outputs = host_outputs
        
        # -----read image
        allocate_place = np.prod(input_data.shape)
        host_inputs[0][:allocate_place] = input_data.flatten(order="C").astype(np.float32)
        context.set_binding_shape(0, input_data.shape)
        # -----inference
        # Transfer input data to the GPU.
        for i in range(len(host_inputs)):
            cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream)
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        # Synchronize the stream
        stream.synchronize()
        # -----get output
        trt_outputs = host_outputs
        
        self.cfx.pop()
        
        return trt_outputs

    def destroy(self):
        self.cfx.pop()