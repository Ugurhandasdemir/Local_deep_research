import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'dart:io';
import 'dart:math';
import 'dart:convert';
import '../main.dart';
import '../models/chat_session.dart';
import '../models/message.dart';
import '../services/api_services.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/settings_drawer.dart';
import '../widgets/welcome.dart';
import 'package:uuid/uuid.dart';
import 'package:file_picker/file_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ChatScreen extends StatefulWidget {
  final bool isDarkMode;
  final ValueChanged<bool> onThemeModeChanged;

  const ChatScreen({
    super.key,
    required this.isDarkMode,
    required this.onThemeModeChanged,
  });

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
  // Search scenarios: fast | balanced | best
  // - fast      -> chromadb + all_mini_l6   (en hizli, kalite orta)
  // - balanced  -> weaviate + bge_squad     (kalite + hiz dengeli)
  // - best      -> milvus + bge_squad       (en yuksek ndcg, olcekli)
  String _searchMode = 'balanced';
  bool _deepResearchEnabled = true;
  static const Map<String, Map<String, String>> _scenarioMap = {
    'fast':     {'db': 'chromadb', 'embedding': 'all_mini_l6'},
    'balanced': {'db': 'weaviate', 'embedding': 'bge_squad'},
    'best':     {'db': 'milvus',   'embedding': 'bge_squad'},
  };
  static const Map<String, String> _scenarioLabel = {
    'fast':     'Hizli',
    'balanced': 'Dengeli',
    'best':     'En Yuksek Basarim',
  };
  static const Map<String, IconData> _scenarioIcon = {
    'fast':     Icons.flash_on,
    'balanced': Icons.balance,
    'best':     Icons.workspace_premium,
  };
  final List<Map<String, dynamic>> _uploadedDocs = [];

  final List<String> _modelOptions = [
    'nemotron-3-nano:4b',
    'medgemma1.5:latest',
    'granite4.1:3b',
    'translategemma:4b',
    'ministral-3:3b',
  ];
  String _selectedModel = '';

  static const String _prefsChatHistoryKey = 'chat_history';

  // ── helpers ──────────────────────────────────────────────────
  bool get isDark => widget.isDarkMode;

  String _normalizedModel(String? value) {
    if (_modelOptions.isEmpty) return '';
    final c = value?.trim();
    if (c == null || c.isEmpty) return _modelOptions.first;
    if (_modelOptions.where((m) => m == c).length == 1) return c;
    return _modelOptions.first;
  }

  List<Message> get _currentMessages {
    if (_currentSessionId == null) return [];
    try {
      return _chatHistory
          .firstWhere((e) => e.id == _currentSessionId)
          .messages;
    } catch (_) {
      return [];
    }
  }

  // ── lifecycle ────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _selectedModel = _normalizedModel(_selectedModel);
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final json = prefs.getString(_prefsChatHistoryKey);
    if (json != null && json.isNotEmpty) {
      try {
        final List<dynamic> decoded = jsonDecode(json);
        setState(() {
          _chatHistory
            ..clear()
            ..addAll(decoded
                .map((e) => ChatSession.fromJson(e as Map<String, dynamic>)));
          _selectedModel = _chatHistory.isNotEmpty
              ? _normalizedModel(_chatHistory.first.model)
              : _normalizedModel(_selectedModel);
        });
      } catch (e) {
        debugPrint('History parse error: $e');
      }
    }
  }

  Future<void> _saveHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(
        _prefsChatHistoryKey,
        jsonEncode(_chatHistory.map((s) => s.toJson()).toList()));
  }

  void _createNewPage() {
    setState(() {
      _currentSessionId = null;
      _isWaitingForResponse = false;
    });
    _saveHistory();
  }

  void _loadSession(String id) {
    setState(() {
      _currentSessionId = id;
      _isWaitingForResponse = false;
      try {
        _selectedModel =
            _normalizedModel(_chatHistory.firstWhere((s) => s.id == id).model);
      } catch (_) {}
    });
    _scrollToBottom();
  }

  void _deleteSession(String id) {
    setState(() {
      _chatHistory.removeWhere((s) => s.id == id);
      if (_currentSessionId == id) _currentSessionId = null;
    });
    _saveHistory();
  }

  // ── PDF upload ───────────────────────────────────────────────
  Future<void> _uploadPdfFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['pdf'],
        allowMultiple: false,
        withData: true,
      );
      if (result == null || result.files.isEmpty) return;
      var bytes = result.files.single.bytes;
      final name = result.files.single.name;
      // Desktop platformlarda bytes null olabilir; dosya yolundan oku
      if (bytes == null) {
        final path = result.files.single.path;
        if (path != null && path.isNotEmpty) {
          bytes = await File(path).readAsBytes();
        }
      }
      if (bytes == null) {
        if (!mounted) return;
        _snack('Dosya verisi okunamadı');
        return;
      }
      if (!mounted) return;
      _snack('PDF işleniyor...', color: AppColors.blue);
      final res = await ApiService.uploadPdfBase64(bytes, name);
      if (!mounted) return;
      if (res['status'] == 'success') {
        _snack('${res['message']} – ${res['pages']} sayfa işlendi',
            color: AppColors.green);
        setState(() {
          _uploadedDocs.insert(0, {
            'name': name,
            'sizeMb': bytes!.length / (1024 * 1024),
            'added': DateTime.now(),
          });
        });
      } else {
        _snack('Hata: ${res['message']}', color: AppColors.red);
      }
    } catch (e) {
      debugPrint('Upload error: $e');
      if (!mounted) return;
      _snack('PDF yüklenirken hata: $e', color: AppColors.red);
    }
  }

  void _snack(String msg, {Color? color}) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg),
      backgroundColor: color,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    ));
  }

  // ── send message ─────────────────────────────────────────────
  Future<void> _handleSendMessage([String? prefill]) async {
    final text = (prefill ?? _textController.text).trim();
    if (text.isEmpty) return;

    final useDeep = _deepResearchEnabled;
    final scenario = _searchMode;
    final scenarioCfg = _scenarioMap[scenario] ?? const {};
    _textController.clear();

    final userMsg = Message(
      id: _uuid.v4(),
      text: text,
      isUser: true,
      timestamp: DateTime.now(),
      responseList: const [],
    );

    setState(() {
      if (_currentSessionId == null) {
        final nid = _uuid.v4();
        _chatHistory.insert(
            0,
            ChatSession(
              id: nid,
              title: text.length > 30 ? '${text.substring(0, 30)}…' : text,
              messages: [userMsg],
              createdAt: DateTime.now(),
              model: _selectedModel,
            ));
        _currentSessionId = nid;
      } else {
        _chatHistory
            .firstWhere((s) => s.id == _currentSessionId)
            .messages
            .add(userMsg);
      }
      _isWaitingForResponse = true;
    });
    _saveHistory();
    _scrollToBottom();

    try {
      Message bot;
      if (useDeep) {
        try {
          final r = await ApiService.getApiResponse(
            text,
            model: _selectedModel,
            scenario: scenario,
            db: scenarioCfg['db'],
            embedding: scenarioCfg['embedding'],
          );
          if (!mounted) return;
          if (r.length < 2 || !r[1]) {
            bot = Message(
                id: _uuid.v4(),
                text: 'Hata: ${r[0]}',
                isUser: false,
                responseList: const []);
          } else {
            bot = Message(
                id: _uuid.v4(),
                text: r[0],
                isUser: false,
                responseList: r[2]);
          }
        } catch (e) {
          bot = Message(
              id: _uuid.v4(),
              text: 'Deep Research başarısız: $e',
              isUser: false,
              responseList: const []);
        }
      } else {
        try {
          final r = await ApiService.normalChat(
            text,
            model: _selectedModel,
            scenario: scenario,
            db: scenarioCfg['db'],
            embedding: scenarioCfg['embedding'],
          );
          if (!mounted) return;
          bot = Message(
              id: _uuid.v4(),
              text: r,
              isUser: false,
              responseList: const []);
        } catch (e) {
          bot = Message(
              id: _uuid.v4(),
              text: 'Chat başarısız: $e',
              isUser: false,
              responseList: const []);
        }
      }

      setState(() {
        _chatHistory
            .firstWhere((s) => s.id == _currentSessionId)
            .messages
            .add(bot);
        _isWaitingForResponse = false;
      });
      _saveHistory();
      _scrollToBottom();
    } catch (e) {
      setState(() {
        _isWaitingForResponse = false;
        _chatHistory
            .firstWhere((s) => s.id == _currentSessionId)
            .messages
            .add(Message(
                id: _uuid.v4(),
                text: 'Beklenmeyen hata: $e',
                isUser: false,
                responseList: const []));
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

  // ══════════════════════════════════════════════════════════════
  //  BUILD
  // ══════════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    final isDesktop = kIsWeb ||
        defaultTargetPlatform == TargetPlatform.linux ||
        defaultTargetPlatform == TargetPlatform.macOS ||
        defaultTargetPlatform == TargetPlatform.windows;
    final hPad =
        isDesktop ? max(200.0, MediaQuery.of(context).size.width * 0.08) : 0.0;

    final bgColor = isDark ? AppColors.darkBg : AppColors.bg;
    final appBarBg = isDark
        ? const Color(0xFF141420).withValues(alpha: 0.92)
        : Colors.white.withValues(alpha: 0.80);

    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: bgColor,
      // ── App bar ──────────────────────────────────────────────
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(60),
        child: ClipRect(
          child: Container(
            decoration: BoxDecoration(
              color: appBarBg,
              border: Border(
                bottom: BorderSide(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.06)
                      : Colors.grey.shade200.withValues(alpha: 0.50),
                ),
              ),
            ),
            child: SafeArea(
              bottom: false,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                child: Row(
                  children: [
                    // hamburger
                    IconButton(
                      icon: Icon(Icons.menu,
                          color: isDark ? Colors.white : AppColors.text,
                          size: 22),
                      onPressed: () =>
                          _scaffoldKey.currentState?.openDrawer(),
                    ),
                    const Spacer(),
                    // model selector pill (centered)
                    _headerModelPill(),
                    const Spacer(),
                    const SizedBox(width: 40), // balance
                  ],
                ),
              ),
            ),
          ),
        ),
      ),

      // ── Drawer ───────────────────────────────────────────────
      drawer: SettingsDrawer(
        isDarkMode: isDark,
        onDarkModeChanged: widget.onThemeModeChanged,
        selectedModel: _selectedModel,
        modelOptions: _modelOptions,
        onModelChanged: (m) => setState(() => _selectedModel = _normalizedModel(m)),
        uploadedDocs: _uploadedDocs,
        onUploadPdf: _uploadPdfFile,
        chatHistory: _chatHistory,
        currentSessionId: _currentSessionId,
        onNewChat: _createNewPage,
        onLoadSession: _loadSession,
        onDeleteSession: _deleteSession,
      ),

      // ── Body ─────────────────────────────────────────────────
      body: Padding(
        padding: EdgeInsets.symmetric(horizontal: hPad),
        child: Stack(
          children: [
            Column(
              children: [
                // model pill below app bar
                if (_currentSessionId != null && _currentMessages.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 6, bottom: 2),
                    child: _usingModelPill(),
                  ),
                // loading
                if (_isWaitingForResponse) _loadingBar(),
                // messages or welcome
                Expanded(
                  child: _currentSessionId == null ||
                          (_currentMessages.isEmpty && !_isWaitingForResponse)
                      ? WelcomeWidget(
                          isDark: isDark,
                          onSuggestionTap: (t) => _handleSendMessage(t),
                        )
                      : ListView.builder(
                          controller: _scrollController,
                          padding:
                              const EdgeInsets.fromLTRB(16, 20, 16, 160),
                          itemCount: _currentMessages.length,
                          itemBuilder: (_, i) => Padding(
                            padding: const EdgeInsets.only(bottom: 20),
                            child: MessageBubble(
                              message: _currentMessages[i],
                              isDark: isDark,
                            ),
                          ),
                        ),
                ),
              ],
            ),

            // ── Input footer ───────────────────────────────────
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: _inputFooter(),
            ),
          ],
        ),
      ),
    );
  }

  // ── Header model pill (PopupMenuButton, centered) ───────────
  Widget _headerModelPill() {
    return PopupMenuButton<String>(
      onSelected: (val) {
        setState(() {
          _selectedModel = _normalizedModel(val);
          if (_currentSessionId != null) {
            final idx =
                _chatHistory.indexWhere((s) => s.id == _currentSessionId);
            if (idx != -1) {
              _chatHistory[idx] =
                  _chatHistory[idx].copyWith(model: _selectedModel);
              _saveHistory();
            }
          }
        });
      },
      offset: const Offset(0, 48),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      color: isDark ? const Color(0xFF1E1E2E) : Colors.white,
      itemBuilder: (_) => _modelOptions.map((m) {
        final isSelected = m == _selectedModel;
        return PopupMenuItem<String>(
          value: m,
          child: Row(
            children: [
              Icon(Icons.smart_toy,
                  size: 18,
                  color: isSelected ? AppColors.purple : Colors.grey),
              const SizedBox(width: 10),
              Text(m,
                  style: TextStyle(
                    fontWeight:
                        isSelected ? FontWeight.w600 : FontWeight.w400,
                    color: isSelected
                        ? AppColors.purple
                        : (isDark ? Colors.white : Colors.grey.shade700),
                  )),
              if (isSelected) ...[
                const Spacer(),
                const Icon(Icons.check, size: 16, color: AppColors.purple),
              ],
            ],
          ),
        );
      }).toList(),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isDark
              ? Colors.white.withValues(alpha: 0.06)
              : Colors.grey.shade50.withValues(alpha: 0.80),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(
            color: isDark
                ? Colors.white.withValues(alpha: 0.08)
                : Colors.grey.shade200,
          ),
          boxShadow: isDark
              ? []
              : [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.04),
                    blurRadius: 6,
                  )
                ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.smart_toy,
                size: 18, color: AppColors.purple),
            const SizedBox(width: 6),
            Text(
              _selectedModel.isEmpty ? 'Select Model' : _selectedModel,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: isDark ? Colors.grey.shade200 : Colors.grey.shade700,
              ),
            ),
            const SizedBox(width: 4),
            Icon(Icons.keyboard_arrow_down,
                size: 16,
                color: isDark ? Colors.grey.shade400 : Colors.grey.shade500),
          ],
        ),
      ),
    );
  }

  // ── "Using: model" pill ──────────────────────────────────────
  Widget _usingModelPill() {
    return Center(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
        decoration: BoxDecoration(
          color: isDark
              ? Colors.white.withValues(alpha: 0.05)
              : Colors.grey.shade100,
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.auto_awesome,
                size: 12, color: AppColors.purple),
            const SizedBox(width: 6),
            Text(
              'Using: $_selectedModel',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: isDark ? Colors.grey.shade400 : Colors.grey.shade500,
                letterSpacing: 0.3,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Loading bar ──────────────────────────────────────────────
  Widget _loadingBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      child: Row(
        children: [
          SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              color: AppColors.blue,
            ),
          ),
          const SizedBox(width: 10),
          Text(
            _deepResearchEnabled
                ? 'PDF Kaynakları Analiz Ediliyor…'
                : '${_scenarioLabel[_searchMode]} modunda arama…',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.blue,
              letterSpacing: 0.3,
            ),
          ),
        ],
      ),
    );
  }

  // ── Floating input footer ────────────────────────────────────
  Widget _inputFooter() {
    final cardBg = isDark
        ? AppColors.darkSurface.withValues(alpha: 0.85)
        : Colors.white.withValues(alpha: 0.80);
    final inputBg = isDark ? AppColors.darkInput : const Color(0xFFF9FAFB);
    final fadeBg = isDark ? AppColors.darkBg : Colors.white;

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [fadeBg, fadeBg, fadeBg.withValues(alpha: 0)],
          stops: const [0.0, 0.80, 1.0],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      child: Container(
        decoration: BoxDecoration(
          color: cardBg,
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: isDark
                ? Colors.white.withValues(alpha: 0.06)
                : Colors.white.withValues(alpha: 0.40),
          ),
          boxShadow: [
            BoxShadow(
              color: const Color(0xFF1F2687).withValues(alpha: 0.07),
              blurRadius: 32,
            ),
          ],
        ),
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Text field
            Container(
              decoration: BoxDecoration(
                color: inputBg,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: isDark
                      ? Colors.white.withValues(alpha: 0.06)
                      : Colors.grey.shade200,
                ),
              ),
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: TextField(
                controller: _textController,
                enabled: !_isWaitingForResponse,
                style: TextStyle(
                  fontSize: 16,
                  color: isDark ? Colors.white : const Color(0xFF1F2937),
                ),
                decoration: InputDecoration(
                  hintText: 'Ask anything…',
                  hintStyle: TextStyle(
                    color: isDark ? Colors.grey.shade500 : Colors.grey.shade400,
                    fontSize: 16,
                  ),
                  border: InputBorder.none,
                  contentPadding:
                      const EdgeInsets.symmetric(vertical: 10),
                ),
                onSubmitted: (_) => _handleSendMessage(),
              ),
            ),
            const SizedBox(height: 10),
            // Bottom row
            Row(
              children: [
                // deep research toggle (button)
                SizedBox(
                  width: 40,
                  height: 22,
                  child: Switch.adaptive(
                    value: _deepResearchEnabled,
                    onChanged: (v) =>
                        setState(() => _deepResearchEnabled = v),
                    activeThumbColor: Colors.white,
                    activeTrackColor: AppColors.blue,
                    inactiveThumbColor: Colors.white,
                    inactiveTrackColor:
                        isDark ? Colors.grey.shade700 : Colors.grey.shade300,
                    materialTapTargetSize:
                        MaterialTapTargetSize.shrinkWrap,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  'Deep Research',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: _deepResearchEnabled
                        ? AppColors.blue
                        : (isDark
                            ? Colors.grey.shade500
                            : Colors.grey.shade500),
                  ),
                ),
                const SizedBox(width: 12),
                // search mode dropdown
                PopupMenuButton<String>(
                  initialValue: _searchMode,
                  tooltip: 'Arama modu',
                  onSelected: (v) => setState(() => _searchMode = v),
                  position: PopupMenuPosition.over,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  itemBuilder: (ctx) => _scenarioLabel.entries.map((e) {
                    final selected = e.key == _searchMode;
                    return PopupMenuItem<String>(
                      value: e.key,
                      child: Row(
                        children: [
                          Icon(
                            _scenarioIcon[e.key],
                            size: 18,
                            color: selected ? AppColors.blue : Colors.grey,
                          ),
                          const SizedBox(width: 10),
                          Text(
                            e.value,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: selected
                                  ? FontWeight.w700
                                  : FontWeight.w500,
                              color: selected ? AppColors.blue : null,
                            ),
                          ),
                          if (selected) ...[
                            const SizedBox(width: 8),
                            Icon(Icons.check, size: 16, color: AppColors.blue),
                          ],
                        ],
                      ),
                    );
                  }).toList(),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 6),
                    decoration: BoxDecoration(
                      color: AppColors.blue.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: AppColors.blue.withValues(alpha: 0.25),
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(_scenarioIcon[_searchMode],
                            size: 16, color: AppColors.blue),
                        const SizedBox(width: 6),
                        Text(
                          _scenarioLabel[_searchMode] ?? 'Mod',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: AppColors.blue,
                          ),
                        ),
                        const SizedBox(width: 4),
                        Icon(Icons.expand_more,
                            size: 16, color: AppColors.blue),
                      ],
                    ),
                  ),
                ),
                const Spacer(),
                // attach
                GestureDetector(
                  onTap: _uploadPdfFile,
                  child: Icon(Icons.attach_file,
                      size: 20,
                      color: isDark
                          ? Colors.grey.shade400
                          : Colors.grey.shade400),
                ),
                const SizedBox(width: 12),
                // send button – gradient
                GestureDetector(
                  onTap:
                      _isWaitingForResponse ? null : () => _handleSendMessage(),
                  child: Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      gradient: _isWaitingForResponse
                          ? null
                          : const LinearGradient(
                              colors: [
                                Color(0xFF6366F1),
                                Color(0xFF9333EA)
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                      color: _isWaitingForResponse
                          ? (isDark
                              ? Colors.grey.shade700
                              : Colors.grey.shade300)
                          : null,
                      shape: BoxShape.circle,
                      boxShadow: _isWaitingForResponse
                          ? []
                          : [
                              BoxShadow(
                                color: const Color(0xFF6366F1)
                                    .withValues(alpha: 0.30),
                                blurRadius: 12,
                                offset: const Offset(0, 4),
                              ),
                            ],
                    ),
                    child: const Icon(Icons.arrow_upward,
                        color: Colors.white, size: 18),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
