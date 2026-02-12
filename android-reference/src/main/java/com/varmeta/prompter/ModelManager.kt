package com.varmeta.prompter

import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ModelManager(private val rootDir: File) {
    suspend fun ensureModelReady(manifestUrl: String, version: String): File = withContext(Dispatchers.IO) {
        if (!rootDir.exists()) {
            rootDir.mkdirs()
        }
        val manifestRaw = fetchText(manifestUrl)
        val manifest = ModelManifest.fromJson(manifestRaw)
        require(manifest.version == version) {
            "Manifest version mismatch: expected=$version actual=${manifest.version}"
        }

        val versionDir = File(rootDir, version)
        if (!versionDir.exists()) {
            versionDir.mkdirs()
        }

        for (file in manifest.files) {
            val dst = File(versionDir, file.path)
            if (dst.exists() && verifyChecksum(dst, file.sha256)) {
                continue
            }
            dst.parentFile?.mkdirs()
            val tmp = File(dst.parentFile, "${dst.name}.part")
            downloadFile(file.url, tmp)
            if (!verifyChecksum(tmp, file.sha256)) {
                tmp.delete()
                throw IllegalStateException("Checksum mismatch for ${file.path}")
            }
            if (dst.exists()) {
                dst.delete()
            }
            if (!tmp.renameTo(dst)) {
                throw IllegalStateException("Failed to move temp file for ${file.path}")
            }
        }
        versionDir
    }

    fun getModelPath(version: String): File = File(rootDir, version)

    fun verifyChecksum(file: File, expectedSha256: String): Boolean {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = stream.read(buffer)
                if (read <= 0) {
                    break
                }
                digest.update(buffer, 0, read)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        return actual.equals(expectedSha256, ignoreCase = true)
    }

    private fun fetchText(url: String): String {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.requestMethod = "GET"
        conn.connectTimeout = 15_000
        conn.readTimeout = 15_000
        conn.inputStream.bufferedReader().use { reader ->
            return reader.readText()
        }
    }

    private fun downloadFile(url: String, dst: File) {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.requestMethod = "GET"
        conn.connectTimeout = 30_000
        conn.readTimeout = 30_000
        conn.inputStream.use { input ->
            FileOutputStream(dst).use { output ->
                input.copyTo(output)
            }
        }
    }
}
