import Foundation
import PDFKit
import Vision
import Speech

actor AttachmentReader {
    enum ReaderError: LocalizedError {
        case unreadable
        case speechPermission
        case offlineSpeechUnavailable

        var errorDescription: String? {
            switch self {
            case .unreadable: return "De bijlage kan niet als tekst worden gelezen."
            case .speechPermission: return "Spraakherkenning heeft geen iOS-toestemming."
            case .offlineSpeechUnavailable: return "Lokale spraakherkenning is niet beschikbaar voor deze taal op dit toestel."
            }
        }
    }

    func text(from url: URL) async throws -> String {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        switch url.pathExtension.lowercased() {
        case "pdf":
            guard let document = PDFDocument(url: url) else { throw ReaderError.unreadable }
            var pages: [String] = []
            for index in 0..<document.pageCount {
                try Task.checkCancellation()
                if let pageText = document.page(at: index)?.string { pages.append(pageText) }
            }
            return pages.joined(separator: "\n\n")
        case "png", "jpg", "jpeg", "heic", "tif", "tiff", "bmp":
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            try VNImageRequestHandler(url: url).perform([request])
            return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
                .joined(separator: "\n")
        case "wav", "m4a", "mp3", "caf", "aiff":
            return try await audioText(from: url, locale: .current)
        default:
            let data = try Data(contentsOf: url)
            if let result = String(data: data, encoding: .utf8) { return result }
            if let result = String(data: data, encoding: .utf16) { return result }
            throw ReaderError.unreadable
        }
    }

    func audioText(from url: URL, locale: Locale) async throws -> String {
        let authorization = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        guard authorization == .authorized else { throw ReaderError.speechPermission }
        guard let recognizer = SFSpeechRecognizer(locale: locale),
              recognizer.supportsOnDeviceRecognition else {
            throw ReaderError.offlineSpeechUnavailable
        }
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = true
        return try await withCheckedThrowingContinuation { continuation in
            var completed = false
            recognizer.recognitionTask(with: request) { result, error in
                guard !completed else { return }
                if let error {
                    completed = true
                    continuation.resume(throwing: error)
                } else if let result, result.isFinal {
                    completed = true
                    continuation.resume(returning: result.bestTranscription.formattedString)
                }
            }
        }
    }
}
