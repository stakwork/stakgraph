import SwiftUI
// @ast node: Class "ChatView"
// @ast node: Class "ChatViewModel"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcesfeatureschatchatviewswift-0"

struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    
    var body: some View {
        List(viewModel.messages, id: \.self) { msg in
            Text(msg)
        }
        .searchable(text: $viewModel.searchText)
    }
}

class ChatViewModel: ObservableObject {
    @Published var messages: [String] = []
    @Published var searchText = ""
}
