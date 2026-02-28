import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../main.dart';
import '../models/chat_session.dart';

class SettingsDrawer extends StatelessWidget {
  // existing chat history parameters are kept for backward compatibility,
  // but the drawer now primarily shows settings and uploaded documents.
  final List<ChatSession> chatHistory;
  final String? currentSessionId;
  final VoidCallback onNewChat;
  final Function(String) onLoadSession;
  final Function(String) onDeleteSession;

  // model selection
  final List<String> modelOptions;
  final String selectedModel;
  final Function(String) onModelChanged;

  // documents uploaded by user
  final List<Map<String, dynamic>> uploadedDocs;

  final VoidCallback onUploadPdf;

  const SettingsDrawer({
    super.key,
    required this.chatHistory,
    required this.currentSessionId,
    required this.onNewChat,
    required this.onLoadSession,
    required this.onDeleteSession,
    required this.modelOptions,
    required this.selectedModel,
    required this.onModelChanged,
    required this.uploadedDocs,
    required this.onUploadPdf,
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      backgroundColor: AppColors.sidebar,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.only(
          topRight: Radius.circular(0),
          bottomRight: Radius.circular(0),
        ),
      ),
      width: 280,
      child: SafeArea(
        child: Column(
          children: [
            // --- Header ---
            Container(
              padding: const EdgeInsets.fromLTRB(24, 20, 16, 20),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(color: Colors.white.withOpacity(0.1)),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                        const Text(
                    "Settings",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      letterSpacing: -0.3,
                    ),
                  ),
                  IconButton(
                    icon: Icon(
                      Icons.close,
                      color: Colors.grey.shade500,
                      size: 20,
                    ),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),

            // --- Main scrollable area with settings and documents ---
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  // model selector section
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 8),
                    child: Text(
                      "AI Model",
                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.5,
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  _buildModelDropdown(context),
                  const SizedBox(height: 24),

                  // uploaded documents section
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 8),
                    child: Text(
                      "Uploaded Documents",
                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.5,
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...uploadedDocs.map((doc) => _buildDocumentTile(context, doc)),

                  const SizedBox(height: 24),

                  // optional chat history at bottom
                  if (chatHistory.isNotEmpty) ...[
                    // new chat button inside history section
                    _buildNewChatButton(context),
                    const SizedBox(height: 20),
                    const Padding(
                      padding: EdgeInsets.symmetric(horizontal: 8),
                      child: Text(
                        "SOHBET GEÇMİŞİ",
                        style: TextStyle(
                          color: Colors.grey,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 1.2,
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    ...chatHistory.map((session) => _buildSessionTile(context, session)),
                  ],
                ],
              ),
            ),

            // --- Upload PDF Button ---
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AppColors.sidebar,
                border: Border(
                  top: BorderSide(color: Colors.white.withOpacity(0.1)),
                ),
              ),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                    onUploadPdf();
                  },
                  icon: const Icon(Icons.cloud_upload_outlined, size: 18),
                  label: const Text(
                    "Upload New PDF",
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.blue,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                    shadowColor: AppColors.blue.withOpacity(0.3),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelDropdown(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
      ),
      child: DropdownButton<String>(
        value: selectedModel,
        dropdownColor: AppColors.sidebar,
        underline: const SizedBox.shrink(),
        isExpanded: true,
        iconEnabledColor: Colors.white,
        style: const TextStyle(color: Colors.white, fontSize: 15),
        items: modelOptions
            .map((m) => DropdownMenuItem(value: m, child: Text(m)))
            .toList(),
        onChanged: (val) {
          if (val != null) onModelChanged(val);
        },
      ),
    );
  }

  Widget _buildDocumentTile(BuildContext context, Map<String, dynamic> doc) {
    final String name = doc['name'] ?? '';
    final double sizeMb = doc['sizeMb'] ?? 0.0;
    final DateTime added = doc['added'] ?? DateTime.now();
    String subtitle = "${sizeMb.toStringAsFixed(1)} MB";
    final diff = DateTime.now().difference(added);
    if (diff.inDays == 0) {
      subtitle += " • Added today";
    } else if (diff.inDays == 1) {
      subtitle += " • Yesterday";
    } else {
      subtitle += " • ${diff.inDays} days ago";
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: GestureDetector(
        onTap: () {},
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.05),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.transparent),
          ),
          child: Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: AppColors.red.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(
                  Icons.picture_as_pdf,
                  color: AppColors.red,
                  size: 16,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      name,
                      style: const TextStyle(
                        color: Colors.grey,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: TextStyle(
                        color: Colors.grey.shade500,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildNewChatButton(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: () {
          onNewChat();
          Navigator.pop(context);
        },
        child: Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.08),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.white.withOpacity(0.1)),
          ),
          child: Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: AppColors.blue.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.add,
                  color: AppColors.blue,
                  size: 18,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                "Yeni Sohbet",
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSessionTile(BuildContext context, ChatSession session) {
    final bool isActive = session.id == currentSessionId;
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(14),
          onTap: () {
            onLoadSession(session.id);
            Navigator.pop(context);
          },
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: isActive
                  ? Colors.white.withOpacity(0.12)
                  : Colors.white.withOpacity(0.04),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: isActive
                    ? Colors.white.withOpacity(0.15)
                    : Colors.transparent,
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                    color: AppColors.purple.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Icon(
                    Icons.chat_bubble_outline,
                    size: 14,
                    color: isActive ? AppColors.blue : Colors.grey.shade400,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        session.title,
                        style: TextStyle(
                          color: Colors.grey.shade200,
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        DateFormat('HH:mm • dd MMM').format(session.createdAt),
                        style: TextStyle(
                          color: Colors.grey.shade600,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: Icon(
                    Icons.delete_outline,
                    size: 16,
                    color: Colors.grey.shade600,
                  ),
                  visualDensity: VisualDensity.compact,
                  padding: EdgeInsets.zero,
                  onPressed: () => onDeleteSession(session.id),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
