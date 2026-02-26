import 'package:flutter/material.dart';
import '../models/chat_session.dart';
import '../models/message.dart';
import '../services/api_services.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/dynamic_row.dart';
import '../widgets/left_drawer.dart';
import '../widgets/welcome.dart';
import 'package:uuid/uuid.dart';
import 'package:file_picker/file_picker.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final Uuid _uuid = const Uuid();
  final TextEditingController _textController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  final List<ChatSession> _chatHistory = [];
  String? _currentSessionId;
  bool _isWaitingForResponse = false;

  String? _selectedTool;

  List<Message> get _currentMessages {
    if (_currentSessionId == null) return [];
    try {
      return _chatHistory
          .firstWhere((element) => element.id == _currentSessionId)
          .messages;
    } catch (e) {
      return [];
    }
  }

  String get _currentTitle {
    if (_currentSessionId == null) return "Yeni Sohbet";
    try {
      return _chatHistory.firstWhere((e) => e.id == _currentSessionId).title;
    } catch (e) {
      return "Hata";
    }
  }


  void _createNewPage() {
    setState(() {
      _currentSessionId = null;
      _isWaitingForResponse = false;
      _selectedTool = null;
    });
  }

  void _loadSession(String sessionId) {
    setState(() {
      _currentSessionId = sessionId;
      _isWaitingForResponse = false;
    });
    _scrollToBottom();
  }

  void _deleteSession(String sessionId) {
    setState(() {
      _chatHistory.removeWhere((s) => s.id == sessionId);
      if (_currentSessionId == sessionId) {
        _currentSessionId = null;
      }
    });
  }

Future<void> _uploadPdfFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        allowMultiple: false,
        withData: true,
      );

      if (result != null && result.files.isNotEmpty) {
        final fileBytes = result.files.single.bytes;
        final fileName = result.files.single.name;
        
        if (fileBytes == null) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Dosya verisi okunamadı")),
          );
          return;
        }
        
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("PDF işleniyor...")),
        );

        final uploadResult = await ApiService.uploadPdfBase64(fileBytes, fileName);

        if (!mounted) return;

        if (uploadResult['status'] == 'success') {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text("${uploadResult['message']}\n${uploadResult['pages']} sayfa işlendi"),
              backgroundColor: Colors.green,
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text("Hata: ${uploadResult['message']}"),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      debugPrint("Hata: $e");
    }
  }
  Future<void> _handleSendMessage() async {
    final text = _textController.text.trim();
    if (text.isEmpty) return;

    final String? activeTool = _selectedTool;
    debugPrint("Mesaj gönderiliyor. Seçili Araç: $activeTool");

    _textController.clear();

    final userMessage = Message(
      id: _uuid.v4(),
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
      const [],
    );

    setState(() {
      if (_currentSessionId == null) {
        final newId = _uuid.v4();
        final newSession = ChatSession(
          id: newId,
          title: text.length > 20 ? "${text.substring(0, 20)}..." : text,
          messages: [userMessage],
          createdAt: DateTime.now(),
        );
        _chatHistory.insert(0, newSession);
        _currentSessionId = newId;
      } else {
        final session = _chatHistory.firstWhere(
          (s) => s.id == _currentSessionId,
        );
        session.messages.add(userMessage);
      }
      _isWaitingForResponse = true;
    });

    _scrollToBottom();

    try {
      late dynamic responseData;
      Message botMessage;
      
      if (activeTool == "Deep Research") {
        try {
          responseData = await ApiService.getApiResponse(text);
          
          if (!mounted) return;
          
          if (responseData.length < 2 || !responseData[1]) {
            botMessage = Message(
              id: _uuid.v4(),
              text: "Hata: ${responseData[0]}",
              isUser: false,
              timestamp: DateTime.now(),
              const [],
            );
          } else {
            debugPrint("responseData[2].toString() ${responseData[2].toString()}");
            botMessage = Message(
              id: _uuid.v4(),
              text: responseData[0],
              isUser: false,
              timestamp: DateTime.now(),
              responseData[2],
            );
          }
        } catch (deepResearchError) {
          debugPrint("Deep Research hatası: $deepResearchError");
          botMessage = Message(
            id: _uuid.v4(),
            text: "Deep Research başarısız: $deepResearchError\n\nLütfen backend sunucusunun çalışıyor olduğundan emin olunuz.",
            isUser: false,
            timestamp: DateTime.now(),
            const [],
          );
        }
      } else {
        debugPrint("normal chat aktif");
        try {
          final responseData = await ApiService.normalChat(text);
          if (!mounted) return;

          botMessage = Message(
            id: _uuid.v4(),
            text: responseData,
            isUser: false,
            timestamp: DateTime.now(),
            const [],
          );
        } catch (normalChatError) {
          debugPrint("Normal Chat hatası: $normalChatError");
          botMessage = Message(
            id: _uuid.v4(),
            text: "Chat başarısız: $normalChatError\n\nLütfen backend sunucusunun çalışıyor olduğundan emin olunuz.",
            isUser: false,
            timestamp: DateTime.now(),
            const [],
          );
        }
      }

      setState(() {
        final session = _chatHistory.firstWhere(
          (s) => s.id == _currentSessionId,
        );
        session.messages.add(botMessage);
        _isWaitingForResponse = false;
      });

      _scrollToBottom();
    } catch (e) {
      debugPrint("Genel hata: $e");
      setState(() {
        _isWaitingForResponse = false;
        
        final botMessage = Message(
          id: _uuid.v4(),
          text: "Beklenmeyen hata: $e",
          isUser: false,
          timestamp: DateTime.now(),
          const [],
        );
        
        final session = _chatHistory.firstWhere(
          (s) => s.id == _currentSessionId,
        );
        session.messages.add(botMessage);
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    debugPrint("Build tetiklendi.");
    return Scaffold(
      appBar: AppBar(
        title: Text(_currentTitle, style: const TextStyle(fontSize: 16)),
        actions: [
          IconButton(
            icon: const Icon(Icons.upload_file),
            onPressed: _uploadPdfFile,
            tooltip: "PDF Yükle",
          ),
        ],
      ),
      drawer: ChatDrawer(
        chatHistory: _chatHistory,
        currentSessionId: _currentSessionId,
        onNewChat: _createNewPage,
        onLoadSession: _loadSession,
        onDeleteSession: _deleteSession,
      ),
      body: Column(
        children: [
          Expanded(
            child:
                _currentSessionId == null ||
                    (_currentMessages.isEmpty && !_isWaitingForResponse)
                ? const WelcomeWidget()
                : ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount:
                        _currentMessages.length +
                        (_isWaitingForResponse ? 1 : 0),
                    itemBuilder: (context, index) {
                      if (index == _currentMessages.length) {
                        return const Center(child: CircularProgressIndicator());
                      }
                      return MessageBubble(message: _currentMessages[index]);
                    },
                  ),
          ),
          Container(
            padding: const EdgeInsets.all(8.0),
            decoration: const BoxDecoration(
              color: Colors.white,
              boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 4)],
            ),
            child: Column(
              children: [
                TextField(
                  controller: _textController,
                  enabled: !_isWaitingForResponse,
                  decoration: const InputDecoration(
                    hintText: "Bir mesaj yazın...",
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(horizontal: 16),
                  ),
                  onSubmitted: (_) => _handleSendMessage(),
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    DynamicMenuRow(
                      selectedValue: _selectedTool,
                      onChanged: (value) {
                        setState(() {
                          _selectedTool = value;
                        });
                      },
                    ),

                    IconButton(
                      icon: Icon(
                        Icons.send_rounded,
                        color: _isWaitingForResponse
                            ? Colors.grey
                            : Colors.indigo,
                      ),
                      onPressed: _isWaitingForResponse
                          ? null
                          : _handleSendMessage,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
