import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../main.dart';
import '../models/message.dart';
import '../screen/pdf_viewer_screen.dart';

class MessageBubble extends StatelessWidget {
  final Message message;
  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return message.isUser ? _buildUserBubble(context) : _buildAiBubble(context);
  }

  // --- User message: light purple bubble, right-aligned, iOS style ---
  Widget _buildUserBubble(BuildContext context) {
    return Align(
      alignment: Alignment.centerRight,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        decoration: BoxDecoration(
          color: AppColors.purpleLight,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
            bottomLeft: Radius.circular(20),
            bottomRight: Radius.circular(4), // iOS-style small corner
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 4,
              offset: const Offset(0, 1),
            ),
          ],
        ),
        child: Text(
          message.text,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 17,
            height: 1.4,
          ),
        ),
      ),
    );
  }

  // --- AI response: plain text with source references ---
  Widget _buildAiBubble(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Status indicator if sources exist
          if (message.responseList.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Row(
                children: [
                  const Icon(
                    Icons.check_circle,
                    size: 14,
                    color: AppColors.statusPurple,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    "Analyzed ${message.responseList.length} PDF Sources",
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: AppColors.statusPurple,
                      letterSpacing: 0.5,
                    ),
                  ),
                ],
              ),
            ),
          // Main AI text
            Container(
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width * 0.92,
              ),
              child: MarkdownBody(
                data: message.text,
                styleSheet: MarkdownStyleSheet(
                  p: const TextStyle(
                    color: AppColors.text,
                    fontSize: 17,
                    height: 1.6,
                  ),
                  strong: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: AppColors.text,
                  ),
                ),
              ),
            ),
          // Source buttons
          if (message.responseList.isNotEmpty) ...[
            const SizedBox(height: 16),
            _buildSourcesRow(context),
          ],
        ],
      ),
    );
  }

  // --- "View Sources" row matching the HTML design ---
  Widget _buildSourcesRow(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // "View Sources" button style header
        GestureDetector(
          onTap: () {
            // nothing for now - user can tap individual items below
          },
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                "View Sources",
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: AppColors.subtext,
                ),
              ),
              const SizedBox(width: 6),
              Row(
                children: List.generate(message.responseList.length, (_) {
                  return Container(
                    margin: const EdgeInsets.only(left: -6),
                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: Colors.grey.shade200),
                    ),
                    child: const Text(
                      "PDF",
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        color: Colors.grey,
                      ),
                    ),
                  );
                }),
              ),
              const SizedBox(width: 4),
              const Icon(
                Icons.chevron_right,
                size: 10,
                color: Colors.grey,
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        // Compact source chips
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: List.generate(message.responseList.length, (index) {
            final item = message.responseList[index];
            final fileName = item["file"] ?? "PDF ${index + 1}";
            return _buildSourceChip(context, index, fileName, item);
          }),
        ),
      ],
    );
  }

  Widget _buildSourceChip(
    BuildContext context,
    int index,
    String fileName,
    dynamic item,
  ) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          String dosyaUrl = item["url"] ?? item["file"] ?? "";
          if (dosyaUrl.isEmpty) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: const Text("PDF URL bulunamadı"),
                behavior: SnackBarBehavior.floating,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            );
            return;
          }
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => PdfViewerScreen(
                pdfDosyaYolu: dosyaUrl,
                isUrl: dosyaUrl.startsWith("http"),
              ),
            ),
          );
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppColors.divider),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.03),
                blurRadius: 4,
                offset: const Offset(0, 1),
              ),
            ],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 28,
                height: 28,
                decoration: BoxDecoration(
                  color: AppColors.red.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(
                  Icons.picture_as_pdf,
                  size: 14,
                  color: AppColors.red,
                ),
              ),
              const SizedBox(width: 8),
              ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 140),
                child: Text(
                  fileName,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: AppColors.text,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 4),
              Text(
                "[${index + 1}]",
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: AppColors.blue,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
