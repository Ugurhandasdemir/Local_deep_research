import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screen/chat_screen.dart';

// ── Design-system colour palette (from code.html) ────────────────────
class AppColors {
  // Primary
  static const Color blue = Color(0xFF6366F1);       // Electric Indigo
  static const Color purple = Color(0xFF9966FF);      // Amethyst
  static const Color purpleDark = Color(0xFF4C1D95);
  static const Color purpleLight = Color(0xFFF3E8FF);
  static const Color purpleVibrant = Color(0xFF8B5CF6);

  // Surfaces – light
  static const Color bg = Color(0xFFFFFFFF);
  static const Color grayInput = Color(0xFFF9FAFB);
  static const Color divider = Color(0xFFE5E7EB);

  // Text – light
  static const Color text = Color(0xFF4B5563);
  static const Color subtext = Color(0xFF9CA3AF);

  // Sidebar
  static const Color sidebar = Color(0xFF1E1B4B);

  // Accents
  static const Color red = Color(0xFFFF3B30);
  static const Color green = Color(0xFF10B981);
  static const Color statusPurple = Color(0xFF6366F1);

  // Dark surfaces
  static const Color darkBg = Color(0xFF0F0F14);
  static const Color darkSurface = Color(0xFF1A1A24);
  static const Color darkCard = Color(0xFF1E1E2A);
  static const Color darkInput = Color(0xFF24243A);

  // Gradients (start, end)
  static const List<Color> purpleGradient = [Color(0xFFA855F7), Color(0xFF7C3AED)];
  static const List<Color> sendGradient = [Color(0xFF6366F1), Color(0xFF9333EA)];
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  static const String _prefsKey = 'theme_mode';
  ThemeMode _themeMode = ThemeMode.light;

  @override
  void initState() {
    super.initState();
    _applyOverlay(_themeMode);
    _loadTheme();
  }

  Future<void> _loadTheme() async {
    final prefs = await SharedPreferences.getInstance();
    final m = prefs.getString(_prefsKey);
    final loaded = m == 'dark' ? ThemeMode.dark : ThemeMode.light;
    if (!mounted) return;
    setState(() => _themeMode = loaded);
    _applyOverlay(loaded);
  }

  Future<void> _setDarkMode(bool enabled) async {
    final next = enabled ? ThemeMode.dark : ThemeMode.light;
    setState(() => _themeMode = next);
    _applyOverlay(next);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKey, enabled ? 'dark' : 'light');
  }

  void _applyOverlay(ThemeMode mode) {
    SystemChrome.setSystemUIOverlayStyle(SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness:
          mode == ThemeMode.dark ? Brightness.light : Brightness.dark,
    ));
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deep Research',
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.light,
        scaffoldBackgroundColor: AppColors.bg,
        colorSchemeSeed: AppColors.blue,
        fontFamily: 'Roboto',
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          foregroundColor: AppColors.text,
          elevation: 0,
          scrolledUnderElevation: 0.5,
          centerTitle: true,
          titleTextStyle: TextStyle(
            color: Color(0xFF1F2937),
            fontSize: 17,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: AppColors.darkBg,
        colorSchemeSeed: AppColors.blue,
        fontFamily: 'Roboto',
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF141420),
          foregroundColor: Colors.white,
          elevation: 0,
          scrolledUnderElevation: 0.5,
          centerTitle: true,
          titleTextStyle: TextStyle(
            color: Colors.white,
            fontSize: 17,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      home: ChatScreen(
        isDarkMode: _themeMode == ThemeMode.dark,
        onThemeModeChanged: _setDarkMode,
      ),
    );
  }
}
