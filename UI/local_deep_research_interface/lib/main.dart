import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'screen/chat_screen.dart';

// iOS-style color palette from design
class AppColors {
  static const Color bg = Color(0xFFF9F9FB);
  static const Color blue = Color(0xFF007AFF);
  static const Color purple = Color(0xFF756AB6);
  static const Color purpleLight = Color(0xFFEDEAF4);
  static const Color grayInput = Color(0xFFF2F2F7);
  static const Color divider = Color(0xFFE5E5EA);
  static const Color red = Color(0xFFFF3B30);
  static const Color text = Color(0xFF1C1C1E);
  static const Color subtext = Color(0xFF8E8E93);
  static const Color sidebar = Color(0xFF1C1C1E);
  static const Color statusPurple = Color(0xFF6B66C5);
}

void main() {
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deep Research',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: AppColors.bg,
        colorSchemeSeed: AppColors.blue,
        brightness: Brightness.light,
        fontFamily: 'Roboto',
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.white,
          foregroundColor: AppColors.text,
          elevation: 0,
          scrolledUnderElevation: 0.5,
          centerTitle: true,
          titleTextStyle: TextStyle(
            color: Colors.black,
            fontSize: 17,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      home: const ChatScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}