import UIKit

class MnistCanvasView: UIView {

    static let SIZE = 28

    var values: [Float]!
    var gridSize: CGFloat!

    func onInit() {
        self.values = [Float].init(repeating: 0, count: MnistCanvasView.SIZE * MnistCanvasView.SIZE)
        self.gridSize = self.frame.width / CGFloat(MnistCanvasView.SIZE)
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        self.onInit()
    }

    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        self.onInit()
    }

    override func draw(_ rect: CGRect) {
        if rect.width <= self.gridSize + 1 {
            let block = UIBezierPath(rect: rect)
            UIColor.setFill(.darkGray)()
            block.fill()
        }
    }

    private func onTouches(_ touches: Set<UITouch>) {
        for touch in touches {
            let loc = touch.location(in: self)
            let x = Int(floor(loc.x / self.gridSize))
            let y = Int(floor(loc.y / self.gridSize))
            if x >= MnistCanvasView.SIZE || y >= MnistCanvasView.SIZE || x < 0 || y < 0 {
                continue
            }

            self.values[MnistCanvasView.SIZE * y + x] = 1
            self.setNeedsDisplay(CGRect(x: self.gridSize * CGFloat(x), y: self.gridSize * CGFloat(y),
                width: self.gridSize, height: self.gridSize))
        }
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.onTouches(touches)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.onTouches(touches)
    }

    func reset() {
        self.values = Array<Float>.init(repeating: 0, count: MnistCanvasView.SIZE * MnistCanvasView.SIZE)
        self.setNeedsDisplay()
    }

}

class MainViewController: UIViewController {
    
    @IBOutlet var labels: [UILabel]!
    @IBOutlet weak var canvas: MnistCanvasView!

    @IBAction func onButtonReset(_ sender: UIButton) {
        self.canvas.reset()
        for label in self.labels {
            label.text = ""
        }
    }

    @IBAction func onButtonRun(_ sender: UIButton) {
        let out = Lenet.getInstance().infer(self.canvas.values)
        for label in self.labels {
            label.text = String(out[label.tag])
        }
    }

}
