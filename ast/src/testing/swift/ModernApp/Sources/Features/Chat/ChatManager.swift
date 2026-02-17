import Foundation

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
