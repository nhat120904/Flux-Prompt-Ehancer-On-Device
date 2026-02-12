import CryptoKit
import Foundation

public actor ModelManager {
    private let rootDirectory: URL
    private let session: URLSession

    public init(rootDirectory: URL, session: URLSession = .shared) {
        self.rootDirectory = rootDirectory
        self.session = session
    }

    public func ensureModelReady(manifestURL: URL, version: String) async throws -> URL {
        try FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        let (manifestData, _) = try await session.data(from: manifestURL)
        let manifest = try JSONDecoder().decode(ModelManifest.self, from: manifestData)
        guard manifest.version == version else {
            throw NSError(domain: "ModelManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Manifest version mismatch"])
        }

        let versionDir = rootDirectory.appendingPathComponent(version, isDirectory: true)
        try FileManager.default.createDirectory(at: versionDir, withIntermediateDirectories: true)

        for file in manifest.files {
            let dst = versionDir.appendingPathComponent(file.path)
            if FileManager.default.fileExists(atPath: dst.path), verifyChecksum(file: dst, expectedSHA256: file.sha256) {
                continue
            }
            try FileManager.default.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            let tmp = dst.deletingPathExtension().appendingPathExtension("part")
            try await downloadFile(urlString: file.url, dst: tmp)
            guard verifyChecksum(file: tmp, expectedSHA256: file.sha256) else {
                try? FileManager.default.removeItem(at: tmp)
                throw NSError(domain: "ModelManager", code: 2, userInfo: [NSLocalizedDescriptionKey: "Checksum mismatch: \(file.path)"])
            }
            if FileManager.default.fileExists(atPath: dst.path) {
                try FileManager.default.removeItem(at: dst)
            }
            try FileManager.default.moveItem(at: tmp, to: dst)
        }

        return versionDir
    }

    public func getModelPath(version: String) -> URL {
        rootDirectory.appendingPathComponent(version, isDirectory: true)
    }

    public func verifyChecksum(file: URL, expectedSHA256: String) -> Bool {
        guard let stream = InputStream(url: file) else { return false }
        stream.open()
        defer { stream.close() }
        var hasher = SHA256()
        var buffer = [UInt8](repeating: 0, count: 1024 * 1024)
        while stream.hasBytesAvailable {
            let read = stream.read(&buffer, maxLength: buffer.count)
            if read < 0 {
                return false
            }
            if read == 0 {
                break
            }
            hasher.update(data: Data(buffer.prefix(read)))
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return digest.caseInsensitiveCompare(expectedSHA256) == .orderedSame
    }

    private func downloadFile(urlString: String, dst: URL) async throws {
        guard let url = URL(string: urlString) else {
            throw NSError(domain: "ModelManager", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])
        }
        let (data, _) = try await session.data(from: url)
        try data.write(to: dst, options: .atomic)
    }
}
