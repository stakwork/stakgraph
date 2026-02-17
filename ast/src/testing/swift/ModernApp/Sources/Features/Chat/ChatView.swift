import SwiftUI

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
