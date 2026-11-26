import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var store: ChatStore

    var body: some View {
        NavigationStack {
            List {
                Section("JJS") {
                    Label("Modellen op deze iPhone: \(store.models.count)", systemImage: "internaldrive")
                    Label("Gesprekken: \(store.chats.count)", systemImage: "bubble.left")
                    if let model = store.models.first(where: { $0.id == store.selectedModelID }) {
                        LabeledContent("Gekozen model", value: model.name)
                    } else {
                        Text("Nog geen model gekozen")
                    }
                }
                Section("Opslag op deze iPhone") {
                    LabeledContent("Beschikbaar", value: formatted(store.storage?.availableBytes))
                    LabeledContent("Totaal", value: formatted(store.storage?.totalBytes))
                    Button("Vernieuwen") { Task { await store.refreshStorage() } }
                }
            }
            .navigationTitle("JJS")
        }
    }

    private func formatted(_ bytes: Int64?) -> String {
        guard let bytes else { return "Onbekend" }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
