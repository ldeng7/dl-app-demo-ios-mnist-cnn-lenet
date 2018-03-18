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

    private func loadVarArray(count: Int) -> [Float] {
        var arr = [Float](repeating: 0, count: count)
        let dataOffsetNext = self.dataOffset + MemoryLayout<Float>.size * count
        _ = self.data.copyBytes(to: UnsafeMutableBufferPointer(start: UnsafeMutablePointer(&arr), count: count),
            from: self.dataOffset ..<  dataOffsetNext)
        self.dataOffset = dataOffsetNext
        return arr
    }

    func loadModel(fromAsset asset: String) {
        let path = Bundle.main.path(forResource: asset, ofType: "dat")
        self.data = try! Data(contentsOf: URL(fileURLWithPath: path!))
        var vars = [[Float]]()
        for i in 0 ..< 8 {
            vars.append(self.loadVarArray(count: Lenet.VARS_COUNT[i]))
        }

        var l = self.addInputLayer(shape: [1, 28, 28])
        l = self.addConvLayer(precedor: l, kh: 5, kw: 5, dOut: 32, s: 1,
            pad: true, weight: vars[0], bias: vars[1])
        l = self.addPoolLayer(precedor: l, isMax: true, k: 2, s: 2, pad: true)
        l = self.addConvLayer(precedor: l, kh: 5, kw: 5, dOut: 64, s: 1,
            pad: true, weight: vars[2], bias: vars[3])
        l = self.addPoolLayer(precedor: l, isMax: true, k: 2, s: 2, pad: true)
        l = self.addFCLayer(precedor: l, szOut: 512, relu: true, weight: vars[4], bias: vars[5])
        l = self.addFCLayer(precedor: l, szOut: 10, relu: false, weight: vars[6], bias: vars[7])
        self.compile()
        self.output = l
    }

    func infer(_ input: [Float]) -> [Float] {
        super.infer(inputs: [input])
        return self.output.out!
    }

}
