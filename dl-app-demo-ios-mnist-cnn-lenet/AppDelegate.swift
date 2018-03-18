import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions lo: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        let lenet = Lenet.getInstance()
        lenet.loadModel(fromAsset: "Assets/cnn_matrix")
        return true
    }

}
