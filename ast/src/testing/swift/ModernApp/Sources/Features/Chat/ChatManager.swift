import Foundation
// @ast node: Class "ChatManager"
// @ast edge: Operand -> Function "send" "ChatManager.swift"
// @ast node: Function "send"
// @ast node: Trait "ChatManagerDelegate"
// @ast node: Class "ConsoleChatObserver"
// @ast edge: Implements -> Trait "ChatManagerDelegate" "ChatManager.swift"
// @ast edge: Operand -> Function "didReceiveMessage" "ChatManager.swift"
// @ast node: Function "didReceiveMessage"
// @ast node: Var "delegate"
// @ast node: Var "onStatusChange"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcesfeatureschatchatmanagerswift-0"

protocol ChatManagerDelegate: AnyObject {
    func didReceiveMessage(_ message: String)
}

class ChatManager {
    weak var delegate: ChatManagerDelegate?
    var onStatusChange: ((Bool) -> Void)?

    func send(message: String) {
        // Simulate network delay
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.delegate?.didReceiveMessage("Echo: \(message)")
            self?.onStatusChange?(true)
        }
    }
}

class ConsoleChatObserver: ChatManagerDelegate {
    func didReceiveMessage(_ message: String) {
        print(message)
    }
}
