import Foundation
import Accelerate

class CNN {
    typealias Layer = (layer: BNNSFilter, szOut: Int)

    static let DATA_TYPE = BNNSDataType.float
    static let INPUT = (w: 28, h: 28)
    static let CONV1 = (w: 5, h: 5, d: 32)
    static let CONV2 = (w: 5, h: 5, d: 64)
    static let FC1_INPUT_SIZE = INPUT.w * INPUT.h * CONV2.d / 16
    static let FC2_INPUT_SIZE = 512
    static let OUTPUT_SIZE = 10

    private static var instance: CNN!

    private var data: Data!
    private var dataOffset = 0
    private var layers = [Layer]()

    private var varConv1Weight = [Float]()
    private var varConv1Bias = [Float]()
    private var varConv2Weight = [Float]()
    private var varConv2Bias = [Float]()
    private var varFC1Weight = [Float]()
    private var varFC1Bias = [Float]()
    private var varFC2Weight = [Float]()
    private var varFC2Bias = [Float]()

    init() {}

    static func getInstance() -> CNN {
        if nil == instance {
            instance = CNN()
        }
        return instance
    }

    private func loadVarArray(arr: inout [Float], count: Int) {
        arr = Array(repeating: 0, count: count)
        let dataOffsetNext = self.dataOffset + MemoryLayout<Float>.size * count
        _ = self.data.copyBytes(to: UnsafeMutableBufferPointer(start: UnsafeMutablePointer(&arr), count: count),
            from: self.dataOffset ..<  dataOffsetNext)
        self.dataOffset = dataOffsetNext
    }

    static private func createConvFilterAndPoolLayer(w: Int, h: Int, dIn: Int, fw: Int, fh: Int, dOut: Int,
            weight: [Float], bias: [Float]) -> [Layer] {
        var inDescFilter = BNNSImageStackDescriptor(
            width: w, height: h, channels: dIn, row_stride: w, image_stride: w * h, data_type: DATA_TYPE
        )
        var outDescFilter = BNNSImageStackDescriptor(
            width: w, height: h, channels: dOut, row_stride: w, image_stride: w * h, data_type: DATA_TYPE
        )
        var layerDescFilter = BNNSConvolutionLayerParameters(
            x_stride: 1, y_stride: 1, x_padding: (fw - 1) / 2, y_padding: (fh - 1) / 2,
            k_width: fw, k_height: fh, in_channels: dIn, out_channels: dOut,
            weights: BNNSLayerData(data: weight, data_type: DATA_TYPE),
            bias: BNNSLayerData(data: bias, data_type: DATA_TYPE),
            activation: BNNSActivation(function: BNNSActivationFunction.rectifiedLinear)
        )
        let szOutFilter = w * h * dOut

        var inDescPool = BNNSImageStackDescriptor(
            width: w, height: h, channels: dOut, row_stride: w, image_stride: w * h, data_type: DATA_TYPE
        )
        var outDescPool = BNNSImageStackDescriptor(
            width: w / 2, height: h / 2, channels: dOut, row_stride: w / 2, image_stride: w * h / 4, data_type: DATA_TYPE
        )
        var layerDescPool = BNNSPoolingLayerParameters(
            x_stride: 2, y_stride: 2, x_padding: 0, y_padding: 0,
            k_width: 2, k_height: 2, in_channels: dOut, out_channels: dOut,
            pooling_function: BNNSPoolingFunction.max
        )
        let szOutPool = w * h * dOut / 4

        return [
            (layer: BNNSFilterCreateConvolutionLayer(&inDescFilter, &outDescFilter, &layerDescFilter, nil)!, szOut: szOutFilter),
            (layer: BNNSFilterCreatePoolingLayer(&inDescPool, &outDescPool, &layerDescPool, nil)!, szOut: szOutPool)
        ]
    }
    
    static private func createFullConnLayer(szIn: Int, szOut: Int, weight: [Float], bias: [Float], isLast: Bool) -> Layer {
        var inDesc = BNNSVectorDescriptor(size: szIn, data_type: DATA_TYPE)
        var outDesc = BNNSVectorDescriptor(size: szOut, data_type: DATA_TYPE)
        var layerDesc = BNNSFullyConnectedLayerParameters(
            in_size: szIn, out_size: szOut,
            weights: BNNSLayerData(data: weight, data_type: DATA_TYPE),
            bias: BNNSLayerData(data: bias, data_type: DATA_TYPE),
            activation: BNNSActivation(function: isLast ?
                BNNSActivationFunction.identity : BNNSActivationFunction.rectifiedLinear)
        )
        return (layer: BNNSFilterCreateFullyConnectedLayer(&inDesc, &outDesc, &layerDesc, nil)!, szOut: szOut)
    }
    
    func loadModel(fromAsset asset: String) {
        let path = Bundle.main.path(forResource: asset, ofType: "dat")
        self.data = try! Data(contentsOf: URL(fileURLWithPath: path!))
        
        self.loadVarArray(arr: &self.varConv1Weight, count: CNN.CONV1.w * CNN.CONV1.h * CNN.CONV1.d)
        self.loadVarArray(arr: &self.varConv1Bias, count: CNN.CONV1.d)
        self.loadVarArray(arr: &self.varConv2Weight, count: CNN.CONV2.w * CNN.CONV2.h * CNN.CONV1.d * CNN.CONV2.d)
        self.loadVarArray(arr: &self.varConv2Bias, count: CNN.CONV2.d)
        self.loadVarArray(arr: &self.varFC1Weight, count: CNN.FC1_INPUT_SIZE * CNN.FC2_INPUT_SIZE)
        self.loadVarArray(arr: &self.varFC1Bias, count: CNN.FC2_INPUT_SIZE)
        self.loadVarArray(arr: &self.varFC2Weight, count: CNN.FC2_INPUT_SIZE * CNN.OUTPUT_SIZE)
        self.loadVarArray(arr: &self.varFC2Bias, count: CNN.OUTPUT_SIZE)
        
        self.layers.append(contentsOf: CNN.createConvFilterAndPoolLayer(
            w: CNN.INPUT.w, h: CNN.INPUT.h, dIn: 1, fw: CNN.CONV1.w, fh: CNN.CONV1.h,
            dOut: CNN.CONV1.d, weight: self.varConv1Weight, bias: self.varConv1Bias
        ))
        self.layers.append(contentsOf: CNN.createConvFilterAndPoolLayer(
            w: CNN.INPUT.w / 2, h: CNN.INPUT.h / 2, dIn: CNN.CONV1.d, fw: CNN.CONV2.w, fh: CNN.CONV2.h,
            dOut: CNN.CONV2.d, weight: self.varConv2Weight, bias: self.varConv2Bias
        ))
        self.layers.append(CNN.createFullConnLayer(
            szIn: CNN.FC1_INPUT_SIZE, szOut: CNN.FC2_INPUT_SIZE, weight: self.varFC1Weight, bias: self.varFC1Bias,
            isLast: false
        ))
        self.layers.append(CNN.createFullConnLayer(
            szIn: CNN.FC2_INPUT_SIZE, szOut: CNN.OUTPUT_SIZE, weight: self.varFC2Weight, bias: self.varFC2Bias,
            isLast: true
        ))
    }
    
    func infer(_ input: [Float]) -> [Float] {
        var arr = input
        for layer in self.layers {
            var o = Array<Float>(repeating: 0, count: layer.szOut)
            BNNSFilterApply(layer.layer, arr, &o)
            arr = o
        }
        return arr
    }
    
}
