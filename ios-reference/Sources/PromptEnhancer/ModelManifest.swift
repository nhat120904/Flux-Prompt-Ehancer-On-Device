import Foundation

public struct ManifestFile: Codable, Sendable {
    public let path: String
    public let sizeBytes: Int
    public let sha256: String
    public let url: String

    enum CodingKeys: String, CodingKey {
        case path
        case sizeBytes = "size_bytes"
        case sha256
        case url
    }
}

public struct ModelManifest: Codable, Sendable {
    public let version: String
    public let minAppVersion: String
    public let files: [ManifestFile]
}
