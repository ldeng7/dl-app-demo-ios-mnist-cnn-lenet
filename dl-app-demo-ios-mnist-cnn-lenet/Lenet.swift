import Foundation
import Accelerate

class Lenet: Network {

    static let VARS_COUNT = [800, 32, 51200, 64, 1605632, 512, 5120, 10]

    private static var instance: Lenet!

    static func getInstance() -> Lenet {
        if nil == instance {
            instance = Lenet()
        }
        return instance
    }

    private var output: NetworkLayer!
    private var data: Data!
    private var dataOffset = 0

    override init() {
        super.init()
    }

    override public func loadModel(fromAsset asset: String) {
        super.loadModel(fromAsset: asset)
        var l = self.addInputLayer(shape: [1, 28, 28])
        l = self.addConvLayer(precedor: l, kh: 5, kw: 5, dOut: 32, s: 1, pad: true)
        l = self.addPoolLayer(precedor: l, isMax: true, k: 2, s: 2, pad: true)
        l = self.addConvLayer(precedor: l, kh: 5, kw: 5, dOut: 64, s: 1, pad: true)
        l = self.addPoolLayer(precedor: l, isMax: true, k: 2, s: 2, pad: true)
        l = self.addFCLayer(precedor: l, szOut: 512, relu: true)
        l = self.addFCLayer(precedor: l, szOut: 10, relu: false)
        self.compile()
        self.output = l
    }

    func infer(_ input: [Float]) -> [Float] {
        super.infer(inputs: [input])
        return self.output.out!
    }

}
