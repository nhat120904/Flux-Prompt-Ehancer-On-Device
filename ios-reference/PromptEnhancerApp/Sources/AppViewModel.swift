import Foundation

@MainActor
final class AppViewModel: ObservableObject {
    @Published var prompt: String = "beautiful house with text 'hello'"
    @Published var output: String = ""
    @Published var status: String = "Ready"
    @Published var isRunning = false
    @Published var latencyMs: Int?
    @Published var tokensGenerated: Int?
    @Published var finishReason: String?

    private let engine = PromptEnhancerEngine()
    private var runningTask: Task<Void, Never>?
    private var warmedUp = false

    func warmup() {
        guard !isRunning else { return }
        isRunning = true
        status = "Warming up model..."

        runningTask = Task {
            do {
                try await engine.warmup()
                await MainActor.run {
                    self.warmedUp = true
                    self.isRunning = false
                    self.status = "Model warmed up"
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.isRunning = false
                    self.status = "Warmup cancelled"
                }
            } catch {
                await MainActor.run {
                    self.isRunning = false
                    self.status = "Warmup failed: \(error.localizedDescription)"
                }
            }
        }
    }

    func enhance() {
        guard !isRunning else { return }

        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            status = "Please input a prompt"
            return
        }

        isRunning = true
        status = "Generating..."
        output = ""
        latencyMs = nil
        tokensGenerated = nil
        finishReason = nil

        runningTask = Task {
            do {
                if !self.warmedUp {
                    try await self.engine.warmup()
                    self.warmedUp = true
                }

                let result = try await self.engine.enhance(prompt: text)
                await MainActor.run {
                    self.output = result.text
                    self.latencyMs = result.latencyMs
                    self.tokensGenerated = result.tokensGenerated
                    self.finishReason = result.finishReason
                    self.isRunning = false
                    self.status = "Done"
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.isRunning = false
                    self.status = "Cancelled"
                }
            } catch {
                await MainActor.run {
                    self.isRunning = false
                    self.status = "Error: \(error.localizedDescription)"
                }
            }
        }
    }

    func cancel() {
        runningTask?.cancel()
        runningTask = nil
    }
}
