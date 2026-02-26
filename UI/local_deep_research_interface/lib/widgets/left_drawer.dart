import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../models/chat_session.dart';

class ChatDrawer extends StatelessWidget {
  final List<ChatSession> chatHistory;
  final String? currentSessionId;
  final VoidCallback onNewChat;
  final Function(String) onLoadSession;
  final Function(String) onDeleteSession;

  const ChatDrawer({
    super.key,
    required this.chatHistory,
    required this.currentSessionId,
    required this.onNewChat,
    required this.onLoadSession,
    required this.onDeleteSession,
  });

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: Column(
        children: [
          SizedBox(height: 20),
          ListTile(
            leading: const Icon(Icons.add),
            title: const Text("Yeni Sohbet"),
            onTap: () {
              onNewChat();
              Navigator.pop(context);
            },
          ),
          const Divider(),
          Expanded(
            child: chatHistory.isEmpty
                ? const Center(child: Text("Geçmiş yok."))
                : ListView.builder(
                    itemCount: chatHistory.length,
                    itemBuilder: (context, index) {
                      final session = chatHistory[index];
                      final bool isActive = session.id == currentSessionId;
                      return ListTile(
                        selected: isActive,
                        selectedTileColor: Colors.indigo.withOpacity(0.1),
                        leading: const Icon(Icons.history),
                        title: Text(
                          session.title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        subtitle: Text(
                          DateFormat('HH:mm').format(session.createdAt),
                        ),
                        trailing: IconButton(
                          icon: const Icon(Icons.delete, size: 18),
                          onPressed: () => onDeleteSession(session.id),
                        ),
                        onTap: () {
                          onLoadSession(session.id);
                          Navigator.pop(context);
                        },
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
