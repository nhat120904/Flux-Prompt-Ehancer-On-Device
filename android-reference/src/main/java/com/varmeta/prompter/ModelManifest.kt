package com.varmeta.prompter

import org.json.JSONArray
import org.json.JSONObject

data class ManifestFile(
    val path: String,
    val sizeBytes: Long,
    val sha256: String,
    val url: String
)

data class ModelManifest(
    val version: String,
    val minAppVersion: String,
    val files: List<ManifestFile>
) {
    companion object {
        fun fromJson(raw: String): ModelManifest {
            val obj = JSONObject(raw)
            val filesJson: JSONArray = obj.getJSONArray("files")
            val files = buildList {
                for (i in 0 until filesJson.length()) {
                    val item = filesJson.getJSONObject(i)
                    add(
                        ManifestFile(
                            path = item.getString("path"),
                            sizeBytes = item.getLong("size_bytes"),
                            sha256 = item.getString("sha256"),
                            url = item.getString("url")
                        )
                    )
                }
            }
            return ModelManifest(
                version = obj.getString("version"),
                minAppVersion = obj.getString("minAppVersion"),
                files = files
            )
        }
    }
}
