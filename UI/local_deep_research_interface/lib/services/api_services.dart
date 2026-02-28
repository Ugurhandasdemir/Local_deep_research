import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

const String API_BASE_URL = 'http://127.0.0.1:8000';

class ApiService {
  static Future<String> normalChat(String message) async {
    try {
      final response = await http.post(
        Uri.parse("$API_BASE_URL/normal/chat"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"input": message}),
      );
      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        return jsonResponse["status"] == "success" ? jsonResponse["summary"] : "Hata: ${jsonResponse['message']}";
      }
      return "Sunucu Hatası: ${response.statusCode}";
    } catch (e) { return "Bağlantı hatası: $e"; }
  }

  static Future<List> getApiResponse(String message) async {
    final url = Uri.parse('$API_BASE_URL/ask/question/ai');
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'input': message}),
      );
      if (response.statusCode == 200) {
        final decodedBody = utf8.decode(response.bodyBytes);
        final jsonResponse = jsonDecode(decodedBody);
        if (jsonResponse['status'] == 'success') {
          return [jsonResponse['aiResponse'] ?? "", true, jsonResponse['sources'] ?? []];
        }
        return [jsonResponse['message'] ?? "Hata oluştu", false];
      }
      return ["Sunucu Hatası: ${response.statusCode}", false];
    } catch (e) { return ["Bağlantı Hatası: $e", false]; }
  }

  static Future<Map<String, dynamic>> uploadPdfBase64(
    Uint8List fileBytes,
    String fileName,
  ) async {
    try {
      final pdfBase64 = base64Encode(fileBytes);
      
      final response = await http.post(
        Uri.parse("$API_BASE_URL/upload/pdf/base64"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "pdf_base64": pdfBase64,
          "pdf_adi": fileName,
        }),
      );

      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        return {
          "status": "success",
          "message": jsonResponse['message'] ?? "PDF işlendi",
          "chunks_added": jsonResponse['chunks_added'] ?? 0,
          "characters": jsonResponse['total_characters'] ?? 0,
          "pages": jsonResponse['pages_processed'] ?? 0,
        };
      }
      return {"status": "error", "message": "HTTP Hatası: ${response.statusCode}"};
    } catch (e) {
      return {"status": "error", "message": "Bağlantı hatası: $e"};
    }
  }
}