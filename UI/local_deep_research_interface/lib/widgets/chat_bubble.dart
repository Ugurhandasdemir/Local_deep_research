import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/message.dart';
import '../screen/pdf_viewer_screen.dart';

class MessageBubble extends StatelessWidget {
  final Message message;
  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.isUser;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: isUser ? Colors.indigo : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              message.text,
              style: TextStyle(
                color: isUser ? Colors.white : Colors.black87,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              DateFormat('HH:mm').format(message.timestamp),
              style: TextStyle(
                color: isUser ? Colors.white70 : Colors.black54,
                fontSize: 10,
              ),
            ),
            isUser
                ? SizedBox()
                : message.responseList.isEmpty
                    ? SizedBox()
                    : SingleChildScrollView(
                        scrollDirection: Axis.horizontal,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.start,
                          children: List.generate(message.responseList.length, (
                            index,
                          ) {
                            final item = message.responseList[index];
                            final fileName = item["file"] ?? "PDF ${index + 1}";
                            return Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 4),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.center,
                                children: [
                                  Tooltip(
                                    message: fileName,
                                    child: Text(
                                      "${index + 1}. pdf",
                                      style: const TextStyle(fontSize: 10),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                  IconButton(
                                    onPressed: () {
                                      debugPrint(
                                          "message.responseList ${message.responseList}");
                                      String dosyaUrl =
                                          item["url"] ?? item["file"] ?? "";
                                      debugPrint("dosya URL = $dosyaUrl");
                                      
                                      if (dosyaUrl.isEmpty) {
                                        ScaffoldMessenger.of(context).showSnackBar(
                                          const SnackBar(
                                              content: Text("PDF URL bulunamadı")),
                                        );
                                        return;
                                      }

                                      Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (context) =>
                                              PdfViewerScreen(
                                                pdfDosyaYolu: dosyaUrl,
                                                isUrl: dosyaUrl
                                                    .startsWith("http"),
                                              ),
                                        ),
                                      );
                                    },
                                    icon: const Icon(Icons.folder_open),
                                    iconSize: 18,
                                  ),
                                ],
                              ),
                            );
                          }),
                        ),
                      ),
          ],
        ),
      ),
    );
  }
}
