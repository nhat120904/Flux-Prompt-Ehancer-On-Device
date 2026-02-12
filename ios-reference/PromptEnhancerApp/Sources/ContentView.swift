import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = AppViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Input Prompt")
                        .font(.headline)

                    TextEditor(text: $viewModel.prompt)
                        .frame(minHeight: 120)
                        .padding(8)
                        .background(Color(.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                    HStack(spacing: 12) {
                        Button("Warmup") {
                            viewModel.warmup()
                        }
                        .buttonStyle(.bordered)
                        .disabled(viewModel.isRunning)

                        Button("Enhance") {
                            viewModel.enhance()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(viewModel.isRunning)

                        Button("Cancel") {
                            viewModel.cancel()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!viewModel.isRunning)
                    }

                    Text("Status: \(viewModel.status)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if let latency = viewModel.latencyMs,
                       let tokens = viewModel.tokensGenerated,
                       let reason = viewModel.finishReason {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Latency: \(latency) ms")
                            Text("Tokens: \(tokens)")
                            Text("Finish reason: \(reason)")
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }

                    Divider()

                    Text("Enhanced Prompt")
                        .font(.headline)

                    Text(viewModel.output.isEmpty ? "(empty)" : viewModel.output)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                        .background(Color(.secondarySystemBackground))
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                }
                .padding()
            }
            .navigationTitle("Prompt Enhancer")
        }
    }
}
