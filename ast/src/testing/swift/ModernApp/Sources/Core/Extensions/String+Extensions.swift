import Foundation
// @ast node: Class "String"
// @ast node: Function "isValidEmail"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcescoreextensionsstringextensionsswift-0"

extension String {
    var localized: String {
        return NSLocalizedString(self, comment: "")
    }
    
    func isValidEmail() -> Bool {
        let emailRegEx = "[A-Z0-9a-z._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,64}"
        let emailPred = NSPredicate(format:"SELF MATCHES %@", emailRegEx)
        return emailPred.evaluate(with: self)
    }
}
