import 'package:flutter/material.dart';
import 'package:pdfrx/pdfrx.dart';

class PdfViewerScreen extends StatelessWidget {
  final String pdfDosyaYolu;
  final bool isUrl;

  const PdfViewerScreen({
    super.key,
    required this.pdfDosyaYolu,
    this.isUrl = false,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(automaticallyImplyLeading: true),
      body: isUrl
          ? PdfViewer.uri(
              Uri.parse(pdfDosyaYolu),
              params: PdfViewerParams(
                minScale: 1.0,
                maxScale: 3.0,
                errorBannerBuilder: (context, error, stackTrace, documentRef) {
                  return Center(child: Text("Hata oluştu: $error"));
                },
              ),
            )
          : PdfViewer.asset(
              "assets/pdfs/$pdfDosyaYolu",
              params: PdfViewerParams(
                minScale: 1.0,
                maxScale: 3.0,
                errorBannerBuilder: (context, error, stackTrace, documentRef) {
                  return Center(child: Text("Hata oluştu: $error"));
                },
              ),
            ),
    );
  }
}
