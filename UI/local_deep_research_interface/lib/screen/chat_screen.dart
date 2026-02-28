import 'package:flutter/material.dart';
import '../main.dart';
import '../models/chat_session.dart';
import '../models/message.dart';
import '../services/api_services.dart';
import '../widgets/chat_bubble.dart';
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
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  final List<ChatSession> _chatHistory = [];
  String? _currentSessionId;
  bool _isWaitingForResponse = false;
  bool _deepResearchEnabled = true;

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

  void _createNewPage() {
    setState(() {
      _currentSessionId = null;
      _isWaitingForResponse = false;
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
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: const Text("Dosya verisi okunamadı"),
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          );
          return;
        }

        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text("PDF işleniyor..."),
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );

        final uploadResult = await ApiService.uploadPdfBase64(fileBytes, fileName);

        if (!mounted) return;

        if (uploadResult['status'] == 'success') {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text("${uploadResult['message']}\n${uploadResult['pages']} sayfa işlendi"),
              backgroundColor: Colors.green.shade600,
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text("Hata: ${uploadResult['message']}"),
              backgroundColor: AppColors.red,
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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

    final bool useDeepResearch = _deepResearchEnabled;
    debugPrint("Mesaj gönderiliyor. Deep Research: $useDeepResearch");

    _textController.clear();

    final userMessage = Message(
      id: _uuid.v4(),
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
      responseList: const [],
    );

    setState(() {
      if (_currentSessionId == null) {
        final newId = _uuid.v4();
        final newSession = ChatSession(
          id: newId,
          title: text.length > 30 ? "${text.substring(0, 30)}..." : text,
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
      Message botMessage;

      if (useDeepResearch) {
        try {
          final responseData = await ApiService.getApiResponse(text);

          if (!mounted) return;

          if (responseData.length < 2 || !responseData[1]) {
            botMessage = Message(
              id: _uuid.v4(),
              text: "Hata: ${responseData[0]}",
              isUser: false,
              timestamp: DateTime.now(),
              responseList: const [],
            );
          } else {
            botMessage = Message(
              id: _uuid.v4(),
              text: responseData[0],
              isUser: false,
              timestamp: DateTime.now(),
              responseList: responseData[2],
            );
          }
        } catch (deepResearchError) {
          debugPrint("Deep Research hatası: $deepResearchError");
          botMessage = Message(
            id: _uuid.v4(),
            text: "Deep Research başarısız: $deepResearchError\n\nLütfen backend sunucusunun çalışıyor olduğundan emin olunuz.",
            isUser: false,
            timestamp: DateTime.now(),
            responseList: const [],
          );
        }
      } else {
        try {
          final responseData = await ApiService.normalChat(text);
          if (!mounted) return;

          botMessage = Message(
            id: _uuid.v4(),
            text: responseData,
            isUser: false,
            timestamp: DateTime.now(),
            responseList: const [],
          );
        } catch (normalChatError) {
          debugPrint("Normal Chat hatası: $normalChatError");
          botMessage = Message(
            id: _uuid.v4(),
            text: "Chat başarısız: $normalChatError\n\nLütfen backend sunucusunun çalışıyor olduğundan emin olunuz.",
            isUser: false,
            timestamp: DateTime.now(),
            responseList: const [],
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
          responseList: const [],
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

  // --- Loading indicator with analyzing text ---
  Widget _buildLoadingIndicator() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 4),
      child: Row(
        children: [
          SizedBox(
            width: 16,
            height: 16,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: AppColors.statusPurple,
            ),
          ),
          const SizedBox(width: 10),
          Text(
            _deepResearchEnabled
                ? "PDF Kaynakları Analiz Ediliyor..."
                : "Yanıt hazırlanıyor...",
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.statusPurple,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: AppColors.bg,
      // --- iOS-style AppBar ---
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(44),
        child: AppBar(
          backgroundColor: Colors.white.withOpacity(0.92),
          elevation: 0,
          scrolledUnderElevation: 0.3,
          centerTitle: true,
          leading: IconButton(
            icon: const Icon(Icons.menu, color: AppColors.blue, size: 24),
            onPressed: () => _scaffoldKey.currentState?.openDrawer(),
          ),
          title: const Text(
            'Deep Research',
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
          ),
          actions: [
            IconButton(
              icon: const Icon(Icons.more_horiz, color: AppColors.blue, size: 24),
              onPressed: () {
                // More options
              },
            ),
          ],
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(0.5),
            child: Container(
              height: 0.5,
              color: Colors.grey.withOpacity(0.2),
            ),
          ),
        ),
      ),
      // --- Dark Sources Drawer ---
      drawer: SourcesDrawer(
        chatHistory: _chatHistory,
        currentSessionId: _currentSessionId,
        onNewChat: _createNewPage,
        onLoadSession: _loadSession,
        onDeleteSession: _deleteSession,
        onUploadPdf: _uploadPdfFile,
      ),
      body: Stack(
        children: [
          // --- Chat messages area ---
          Column(
            children: [
              Expanded(
                child: _currentSessionId == null ||
                        (_currentMessages.isEmpty && !_isWaitingForResponse)
                    ? const WelcomeWidget()
                    : ListView.builder(
                        controller: _scrollController,
                        padding: const EdgeInsets.fromLTRB(16, 24, 16, 140),
                        itemCount: _currentMessages.length +
                            (_isWaitingForResponse ? 1 : 0),
                        itemBuilder: (context, index) {
                          if (index == _currentMessages.length) {
                            return _buildLoadingIndicator();
                          }
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 16),
                            child: MessageBubble(
                              message: _currentMessages[index],
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
          // --- Floating Input Footer ---
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [
                    Colors.white,
                    Colors.white,
                    Colors.white.withOpacity(0),
                  ],
                  stops: const [0.0, 0.85, 1.0],
                ),
              ),
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.08),
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                  border: Border.all(color: Colors.grey.shade100),
                ),
                padding: const EdgeInsets.all(16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // --- Text Input ---
                    Container(
                      decoration: BoxDecoration(
                        color: AppColors.grayInput,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                      child: TextField(
                        controller: _textController,
                        enabled: !_isWaitingForResponse,
                        style: const TextStyle(
                          fontSize: 17,
                          color: AppColors.text,
                        ),
                        decoration: const InputDecoration(
                          hintText: "Bir şey sorun...",
                          hintStyle: TextStyle(
                            color: AppColors.subtext,
                            fontSize: 17,
                          ),
                          border: InputBorder.none,
                          contentPadding: EdgeInsets.symmetric(vertical: 10),
                        ),
                        onSubmitted: (_) => _handleSendMessage(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    // --- Bottom row: Toggle + Actions ---
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        // Deep Research Toggle
                        Row(
                          children: [
                            SizedBox(
                              width: 44,
                              height: 24,
                              child: Switch(
                                value: _deepResearchEnabled,
                                onChanged: (val) {
                                  setState(() {
                                    _deepResearchEnabled = val;
                                  });
                                },
                                activeColor: Colors.white,
                                activeTrackColor: AppColors.blue,
                                inactiveThumbColor: Colors.white,
                                inactiveTrackColor: Colors.grey.shade300,
                                materialTapTargetSize:
                                    MaterialTapTargetSize.shrinkWrap,
                              ),
                            ),
                            const SizedBox(width: 10),
                            const Text(
                              'Deep Research',
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w600,
                                color: Color(0xFF333333),
                              ),
                            ),
                          ],
                        ),
                        // Attach + Send
                        Row(
                          children: [
                            // Paperclip / Attach
                            GestureDetector(
                              onTap: _uploadPdfFile,
                              child: Transform.rotate(
                                angle: 0.785398, // 45 degrees
                                child: Icon(
                                  Icons.attach_file,
                                  color: Colors.grey.shade400,
                                  size: 22,
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            // Send Button
                            GestureDetector(
                              onTap: _isWaitingForResponse
                                  ? null
                                  : _handleSendMessage,
                              child: Container(
                                width: 36,
                                height: 36,
                                decoration: BoxDecoration(
                                  color: _isWaitingForResponse
                                      ? Colors.grey.shade300
                                      : AppColors.blue,
                                  shape: BoxShape.circle,
                                  boxShadow: _isWaitingForResponse
                                      ? []
                                      : [
                                          BoxShadow(
                                            color: AppColors.blue
                                                .withOpacity(0.3),
                                            blurRadius: 8,
                                            offset: const Offset(0, 2),
                                          ),
                                        ],
                                ),
                                child: const Icon(
                                  Icons.arrow_upward,
                                  color: Colors.white,
                                  size: 18,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
