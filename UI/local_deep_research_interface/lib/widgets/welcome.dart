import 'package:flutter/material.dart';
import '../main.dart';

class WelcomeWidget extends StatelessWidget {
  final bool isDark;
  final ValueChanged<String>? onSuggestionTap;

  const WelcomeWidget({
    super.key,
    this.isDark = false,
    this.onSuggestionTap,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Gradient icon
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    AppColors.purple.withValues(alpha: 0.18),
                    AppColors.blue.withValues(alpha: 0.12),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(24),
              ),
              child: const Icon(
                Icons.auto_awesome,
                size: 36,
                color: AppColors.purple,
              ),
            ),
            const SizedBox(height: 24),
            Text(
              'Deep Research',
              style: TextStyle(
                fontSize: 26,
                fontWeight: FontWeight.bold,
                color: isDark ? Colors.white : const Color(0xFF1F2937),
                letterSpacing: -0.5,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              'PDF dosyalarınızı yükleyin ve sorularınızı sorun.\nYapay zeka kaynaklarınızı analiz edecek.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 15,
                color: isDark ? Colors.grey.shade400 : AppColors.subtext,
                height: 1.6,
              ),
            ),
            const SizedBox(height: 32),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              alignment: WrapAlignment.center,
              children: [
                _chip('Quantum computing nedir?'),
                _chip('PDF özetini çıkar'),
                _chip('Temel kavramları açıkla'),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _chip(String text) {
    return GestureDetector(
      onTap: onSuggestionTap != null ? () => onSuggestionTap!(text) : null,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        decoration: BoxDecoration(
          color: isDark ? AppColors.darkCard : Colors.white,
          borderRadius: BorderRadius.circular(22),
          border: Border.all(
            color: isDark
                ? Colors.white.withValues(alpha: 0.08)
                : AppColors.divider,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: isDark ? 0.12 : 0.04),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Text(
          text,
          style: TextStyle(
            fontSize: 14,
            color: isDark ? Colors.grey.shade300 : AppColors.subtext,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }
}
