import 'package:flutter/material.dart';

class DynamicMenuRow extends StatelessWidget {
  final String? selectedValue;
  final ValueChanged<String?> onChanged;

  const DynamicMenuRow({
    super.key,
    required this.selectedValue,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: const BoxConstraints(maxWidth: 200), 
      padding: const EdgeInsets.symmetric(horizontal: 8.0),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (selectedValue == null)
            const Text(
              "Araçlar",
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500, color: Colors.grey),
            )
          else
            Flexible(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.auto_awesome, color: Colors.indigo, size: 18),
                  const SizedBox(width: 4),
                  Flexible(
                    child: Text(
                      selectedValue!,
                      style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),

          selectedValue == null
              ? PopupMenuButton<String>(
                  icon: const Icon(Icons.add_circle_outline, color: Colors.indigo),
                  onSelected: (value) => onChanged(value),
                  itemBuilder: (context) => [
                    const PopupMenuItem(
                      value: "Deep Research",
                      child: Text("Deep Research"),
                    ),
                  ],
                )
              : IconButton(
                  icon: const Icon(Icons.cancel, color: Colors.red, size: 20),
                  onPressed: () => onChanged(null),
                ),
        ],
      ),
    );
  }
}