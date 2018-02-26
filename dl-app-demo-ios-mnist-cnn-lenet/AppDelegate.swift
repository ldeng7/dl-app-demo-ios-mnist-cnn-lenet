import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions lo: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        let cnn = CNN.getInstance()
        cnn.loadModel(fromAsset: "Assets/cnn_matrix")
        return true
    }

}
