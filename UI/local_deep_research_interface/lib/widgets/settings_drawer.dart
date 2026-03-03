import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../main.dart';
import '../models/chat_session.dart';

class SettingsDrawer extends StatelessWidget {
  final List<ChatSession> chatHistory;
  final String? currentSessionId;
  final VoidCallback onNewChat;
  final Function(String) onLoadSession;
  final Function(String) onDeleteSession;

  final List<String> modelOptions;
  final String selectedModel;
  final Function(String) onModelChanged;

  final bool isDarkMode;
  final ValueChanged<bool> onDarkModeChanged;

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
    required this.isDarkMode,
    required this.onDarkModeChanged,
    required this.uploadedDocs,
    required this.onUploadPdf,
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      backgroundColor: AppColors.sidebar,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
      width: 280,
      child: SafeArea(
        child: Column(
          children: [
            // ── Header ──────────────────────────────────────────
            Container(
              padding: const EdgeInsets.fromLTRB(24, 20, 16, 20),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // gradient title
                  ShaderMask(
                    shaderCallback: (bounds) => const LinearGradient(
                      colors: [Color(0xFFA5B4FC), Color(0xFFD8B4FE)],
                    ).createShader(bounds),
                    child: const Text(
                      'History',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        letterSpacing: -0.3,
                      ),
                    ),
                  ),
                  IconButton(
                    icon: Icon(Icons.close,
                        color: Colors.grey.shade500, size: 20),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),

            // ── Scrollable body ─────────────────────────────────
            Expanded(
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  // Dark mode toggle
                  _darkModeRow(),
                  const SizedBox(height: 20),

                  // AI Model selector
                  _sectionLabel('AI Model'),
                  const SizedBox(height: 8),
                  _modelSelector(context),
                  const SizedBox(height: 24),

                  // Uploaded documents
                  if (uploadedDocs.isNotEmpty) ...[
                    _sectionLabel('Uploaded Documents'),
                    const SizedBox(height: 8),
                    ...uploadedDocs.map((d) => _docTile(d)),
                    const SizedBox(height: 24),
                  ],

                  // Chat history
                  if (chatHistory.isNotEmpty) ...[
                    _sectionLabel('Previous Chats'),
                    const SizedBox(height: 10),
                    ...chatHistory.map((s) => _sessionTile(context, s)),
                  ],
                ],
              ),
            ),

            // ── Bottom buttons ───────────────────────────────
            Container(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
              decoration: BoxDecoration(
                color: AppColors.sidebar,
                border: Border(
                  top: BorderSide(color: Colors.white.withValues(alpha: 0.05)),
                ),
              ),
              child: Column(
                children: [
                  // Upload PDF
                  SizedBox(
                    width: double.infinity,
                    child: OutlinedButton.icon(
                      onPressed: () {
                        Navigator.pop(context);
                        onUploadPdf();
                      },
                      icon: const Icon(Icons.cloud_upload_outlined, size: 16),
                      label: const Text('Upload PDF',
                          style: TextStyle(fontWeight: FontWeight.w600)),
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.white,
                        side: BorderSide(
                            color: Colors.white.withValues(alpha: 0.12)),
                        padding: const EdgeInsets.symmetric(vertical: 13),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  // New chat
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        onNewChat();
                        Navigator.pop(context);
                      },
                      icon: const Icon(Icons.add, size: 18),
                      label: const Text('New Chat',
                          style: TextStyle(fontWeight: FontWeight.w600)),
                      style: ElevatedButton.styleFrom(
                        backgroundColor:
                            Colors.white.withValues(alpha: 0.10),
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(vertical: 13),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Reusable pieces ─────────────────────────────────────────────

  Widget _sectionLabel(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Text(
        text.toUpperCase(),
        style: TextStyle(
          color: const Color(0xFFA5B4FC).withValues(alpha: 0.50),
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 1.5,
        ),
      ),
    );
  }

  Widget _darkModeRow() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.05)),
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: const Color(0xFF6366F1).withValues(alpha: 0.20),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(Icons.dark_mode,
                size: 16, color: Color(0xFFA5B4FC)),
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text('Dark Mode',
                style: TextStyle(
                    color: Colors.white70,
                    fontSize: 14,
                    fontWeight: FontWeight.w500)),
          ),
          SizedBox(
            height: 28,
            child: Switch.adaptive(
              value: isDarkMode,
              onChanged: onDarkModeChanged,
              activeThumbColor: AppColors.blue,
              activeTrackColor: AppColors.blue.withValues(alpha: 0.40),
            ),
          ),
        ],
      ),
    );
  }

  Widget _modelSelector(BuildContext context) {
    final uniqueOptions = modelOptions.toSet().toList();
    final safe =
        uniqueOptions.contains(selectedModel) ? selectedModel : null;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
      ),
      child: DropdownButton<String>(
        value: safe,
        hint: Text(
          uniqueOptions.isNotEmpty ? uniqueOptions.first : 'No model',
          style: const TextStyle(color: Colors.white, fontSize: 14),
        ),
        dropdownColor: const Color(0xFF2C2C2E),
        underline: const SizedBox.shrink(),
        isExpanded: true,
        iconEnabledColor: Colors.grey.shade400,
        style: const TextStyle(color: Colors.white, fontSize: 14),
        borderRadius: BorderRadius.circular(12),
        items: uniqueOptions
            .map((m) => DropdownMenuItem(
                  value: m,
                  child: Text(m,
                      style:
                          TextStyle(color: m == safe ? AppColors.blue : null)),
                ))
            .toList(),
        onChanged: (v) {
          if (v != null) onModelChanged(v);
        },
      ),
    );
  }

  Widget _docTile(Map<String, dynamic> doc) {
    final name = doc['name'] ?? '';
    final mb = (doc['sizeMb'] ?? 0.0) as double;
    final added = doc['added'] ?? DateTime.now();
    final diff = DateTime.now().difference(added as DateTime);
    String sub = '${mb.toStringAsFixed(1)} MB';
    if (diff.inDays == 0) {
      sub += ' • Added today';
    } else if (diff.inDays == 1) {
      sub += ' • Yesterday';
    } else {
      sub += ' • ${diff.inDays} days ago';
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: AppColors.red.withValues(alpha: 0.20),
                borderRadius: BorderRadius.circular(8),
              ),
              child:
                  const Icon(Icons.picture_as_pdf, size: 16, color: AppColors.red),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name,
                      style: const TextStyle(
                          color: Colors.grey,
                          fontSize: 13,
                          fontWeight: FontWeight.w500),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 2),
                  Text(sub,
                      style: TextStyle(
                          color: Colors.grey.shade600, fontSize: 11)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _sessionTile(BuildContext context, ChatSession session) {
    final isActive = session.id == currentSessionId;
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () {
            onLoadSession(session.id);
            Navigator.pop(context);
          },
          child: Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: isActive
                  ? Colors.white.withValues(alpha: 0.10)
                  : Colors.white.withValues(alpha: 0.04),
              borderRadius: BorderRadius.circular(12),
              border: isActive
                  ? const Border(
                      left: BorderSide(color: Color(0xFF818CF8), width: 2))
                  : null,
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        session.title,
                        style: TextStyle(
                          color: isActive ? Colors.white : Colors.grey.shade300,
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        DateFormat('HH:mm • dd MMM')
                            .format(session.createdAt),
                        style: TextStyle(
                          color: isActive
                              ? const Color(0xFFA5B4FC)
                              : Colors.grey.shade600,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.delete_outline,
                      size: 14, color: Colors.grey.shade600),
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
