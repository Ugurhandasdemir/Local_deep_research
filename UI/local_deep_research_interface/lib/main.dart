import 'package:flutter/material.dart';
import 'screen/chat_screen.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Sohbet Asistanı',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: ChatScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}