import Foundation
import Accelerate

public class NetworkLayer {

    var filter: BNNSFilter!
    var precedor: NetworkLayer!
    var successors: [NetworkLayer] = []
    var szOut: Int = 0
    var shapeOut: [Int]!

    var level: Int = 0
    public var out: [Float]!

    init() {}

    func infer() {
        var out = [Float](repeating: 0, count: self.szOut)
        BNNSFilterApply(self.filter, self.precedor.out, &out)
        self.out = out
    }

}

class InputLayer: NetworkLayer {

    override init() {
        super.init()
        self.level = 1
    }

    override func infer() {}

}

class ConcatLayer: NetworkLayer {

    var precedors: [NetworkLayer]!

    override init() {
        super.init()
    }

    override func infer() {
        self.out = self.precedors.reduce([Float](), {$0 + $1.out})
    }

}

public class Network {

    private var compiled: Bool = false
    private var inputs: [NetworkLayer] = []
    private var layers: [NetworkLayer] = []
    private var levels: [[NetworkLayer]] = []

    public init() {}

    public func addInputLayer(shape: [Int]) -> NetworkLayer {
        let l = InputLayer()
        self.layers.append(l)
        self.inputs.append(l)
        l.szOut = shape.reduce(1, {$0 * $1})
        l.shapeOut = shape
        return l
    }

    public func addFCLayer(precedor: NetworkLayer, szOut: Int, relu: Bool,
            weight: [Float], bias: [Float]) -> NetworkLayer {
        let l = NetworkLayer()
        self.layers.append(l)
        precedor.successors.append(l)
        l.precedor = precedor

        let szIn = precedor.szOut
        var inDesc = BNNSVectorDescriptor(size: szIn, data_type: BNNSDataType.float)
        var outDesc = BNNSVectorDescriptor(size: szOut, data_type: BNNSDataType.float)
        var layerDesc = BNNSFullyConnectedLayerParameters(
            in_size: szIn,
            out_size: szOut,
            weights: BNNSLayerData(data: weight, data_type: BNNSDataType.float),
            bias: BNNSLayerData(data: bias, data_type: BNNSDataType.float),
            activation: BNNSActivation(function: relu ?
                BNNSActivationFunction.rectifiedLinear : BNNSActivationFunction.identity)
        )

        l.filter = BNNSFilterCreateFullyConnectedLayer(&inDesc, &outDesc, &layerDesc, nil)
        l.szOut = szOut
        l.shapeOut = [szOut]
        return l
    }

    static private func calcConvSize(l: Int, k: Int, s: Int, pad: Bool) -> (o: Int, p: Int) {
        let fl = Float(l)
        let fk = Float(k)
        let fs = Float(s)
        var fo: Float
        var p: Int
        if pad {
            fo = ceil(fl / fs)
            p = (Int(fo) - 1) * s + k - l
            p += p & 1
            p >>= 1
        } else {
            fo = ceil((fl - fk + 1) / fs)
            p = 0
        }
        return (o: Int(fo), p: p)
    }

    public func addConvLayer(precedor: NetworkLayer, kh: Int, kw: Int, dOut: Int, s: Int,
            pad: Bool, weight: [Float], bias: [Float],
            act: BNNSActivationFunction = BNNSActivationFunction.rectifiedLinear) -> NetworkLayer {
        let l = NetworkLayer()
        self.layers.append(l)
        precedor.successors.append(l)
        l.precedor = precedor

        let dIn = precedor.shapeOut[0]
        let h = precedor.shapeOut[1]
        let w = precedor.shapeOut[2]
        let szh = Network.calcConvSize(l: h, k: kh, s: s, pad: pad)
        let szw = Network.calcConvSize(l: w, k: kw, s: s, pad: pad)

        var inDesc = BNNSImageStackDescriptor(
            width: w,
            height: h,
            channels: dIn,
            row_stride: w,
            image_stride: w * h,
            data_type: BNNSDataType.float
        )
        var outDesc = BNNSImageStackDescriptor(
            width: szw.o,
            height: szh.o,
            channels: dOut,
            row_stride: szw.o,
            image_stride: szw.o * szh.o,
            data_type: BNNSDataType.float
        )
        var layerDesc = BNNSConvolutionLayerParameters(
            x_stride: s,
            y_stride: s,
            x_padding: szw.p,
            y_padding: szh.p,
            k_width: kw,
            k_height: kh,
            in_channels: dIn,
            out_channels: dOut,
            weights: BNNSLayerData(data: weight, data_type: BNNSDataType.float),
            bias: BNNSLayerData(data: bias, data_type: BNNSDataType.float),
            activation: BNNSActivation(function: act)
        )

        l.filter = BNNSFilterCreateConvolutionLayer(&inDesc, &outDesc, &layerDesc, nil)
        l.szOut = szw.o * szh.o * dOut
        l.shapeOut = [dOut, szh.o, szw.o]
        return l
    }

    public func addPoolLayer(precedor: NetworkLayer, isMax: Bool, k: Int, s: Int, pad: Bool) -> NetworkLayer {
        let l = NetworkLayer()
        self.layers.append(l)
        precedor.successors.append(l)
        l.precedor = precedor

        let d = precedor.shapeOut[0]
        let h = precedor.shapeOut[1]
        let w = precedor.shapeOut[2]
        let szh = Network.calcConvSize(l: h, k: k, s: s, pad: pad)
        let szw = Network.calcConvSize(l: w, k: k, s: s, pad: pad)

        var inDesc = BNNSImageStackDescriptor(
            width: w,
            height: h,
            channels: d,
            row_stride: w,
            image_stride: w * h,
            data_type: BNNSDataType.float
        )
        var outDesc = BNNSImageStackDescriptor(
            width: szw.o,
            height: szh.o,
            channels: d,
            row_stride: szw.o,
            image_stride: szw.o * szh.o,
            data_type: BNNSDataType.float
        )
        var layerDesc = BNNSPoolingLayerParameters(
            x_stride: s,
            y_stride: s,
            x_padding: szw.p,
            y_padding: szh.p,
            k_width: k,
            k_height: k,
            in_channels: d,
            out_channels: d,
            pooling_function: isMax ? BNNSPoolingFunction.max : BNNSPoolingFunction.average
        )

        l.filter = BNNSFilterCreatePoolingLayer(&inDesc, &outDesc, &layerDesc, nil)
        l.szOut = szw.o * szh.o * d
        l.shapeOut = [d, szh.o, szw.o]
        return l
    }

    public func addConcatLayer(precedors: [NetworkLayer]) -> NetworkLayer {
        let l = ConcatLayer()
        self.layers.append(l)
        for precedor in precedors {
            precedor.successors.append(l)
        }
        l.precedors = precedors

        let h = precedors[0].shapeOut[1]
        let w = precedors[0].shapeOut[2]
        let d = precedors.reduce(0, {$0 + $1.shapeOut[0]})
        l.szOut = w * h * d
        l.shapeOut = [d, h, w]
        return l
    }

    public func compile() {
        var levelNext = 1
        var layersNext = self.inputs
        repeat {
            let layersCur = layersNext
            layersNext = []
            levelNext += 1
            for layer in layersCur {
                for suc in layer.successors {
                    suc.level = levelNext
                    layersNext.append(suc)
                }
            }
        } while layersNext.count > 0

        self.levels = [[NetworkLayer]](repeating: [], count: levelNext - 1)
        for layer in self.layers {
            if layer.level >= 1 {
                self.levels[layer.level - 1].append(layer)
            }
        }
        self.compiled = true
    }

    public func infer(inputs: [[Float]]) {
        if !self.compiled {
            self.compile()
        }
        for i in 0 ..< self.inputs.count {
            self.inputs[i].out = inputs[i]
        }
        for level in self.levels {
            for layer in level {
                layer.infer()
            }
        }
    }

}
