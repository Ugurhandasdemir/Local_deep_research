# Ortak çalışma protokolü (insanlar ve AI asistanları için)

Bu repoda üç kişi makale yazıyor: Uğurhan Daşdemir (`Ugurhandasdemir`), Yüksel Erhan Turgut (`Yukseltt`)
ve Hakan Enes Demir (`hakanenesdemir`). Her birimiz kendi AI asistanıyla çalışıyor (Claude, Codex, Gemini...).
Sen bu kişilerden birinin asistanıysan bu dosya sana yazıldı. Kökteki `AGENTS.md` ve `CLAUDE.md` seni buraya yönlendirir.

## 1. Kim için çalıştığını belirt

Hangi kişi için çalıştığını bilmiyorsan, herhangi bir dosyaya dokunmadan önce kullanıcına sor.
Eklediğin her maddede ve commit'te o kişinin adını ve kullandığın aracı yaz, örnek: `ekleyen: Hakan (Codex)`.

## 2. Yapılacaklar listesi (kökteki `README.md`)

- README'de yalnızca yapılacaklar listesi var. Başka içerik ekleme, eski README `docs/` altında yedekli.
- Yeni madde eklemek serbest. Biçim README'nin başında yazıyor. Yeni madde en son kullanılan numaradan devam eder (`Y-29`, `Y-30`...).
- **Başkasının eklediği maddeyi silme, metnini değiştirme, numarasını değiştirme.** Katılmıyorsan GitHub issue aç (bölüm 4).
- Bir maddeyi üstlenmek için `sorumlu: —` alanına kendi adını yaz. Başkası üstlenmişse o kişiye sormadan değiştirme.
- İş bitince kutuyu işaretle (`- [x]`) ve satırın sonuna PR ya da commit linkini ekle. Maddeyi silme.
- Her maddenin bir kaynağı olsun: dosya:satır, issue, incelemeler/ altındaki bir not ya da makaledeki bölüm. Kaynağı olmayan iddia ekleme.
- Ölçülmemiş bir sonucu "bulduk/kanıtladık" diye yazma. Tahminse açıkça tahmin olduğunu belirt.

## 3. Değişiklik akışı

- Herkes kendi dalında çalışır: `ugur`, `yuksel`, `hakan`. Bu dallar main'in gerisinde kaldı. Başlamadan önce `git fetch && git checkout <dal> && git merge origin/main` komutlarını çalıştır.
- main'e doğrudan push yapma. Kendi dalından PR aç, PR açıklamasında hangi `Y-NN` maddelerine dokunduğunu yaz.
- Yalnızca README'ye madde ekleyen küçük PR'lar da olur; diğer kişilerden biri baktıktan sonra merge edilir.
- Büyük dosya (>10 MB), veri seti, model ağırlığı, `DB/` içeriği ya da kişisel ayar dosyası (`.claude/`, `.vscode/`, `.env`) commit'leme.
- Makale taslağı ve bölüm notları `makale/` altına gider. İnceleme ve değerlendirme notları `makale/incelemeler/YYYY-AA-GG-<ad>-<araç>.md` adıyla eklenir.

## 4. Anlaşmazlık ve tartışma: GitHub issue

- Bir maddeye, bir PR'a ya da makaledeki bir iddiaya itirazın varsa issue aç ve `tartışma` etiketini koy.
- Başlık biçimi: `[Y-NN] kısa konu`. Gövdede ne yazıldığını, neden sorun olduğunu ve kaynağını yaz, sonra öneriyi ekle.
- Karar insanlarındır. AI issue açabilir, yorum yazabilir; issue kapatma ve "karar verildi" yazma işini kişiler yapar.
- Karar çıkınca ilgili madde güncellenir ve satırın sonuna issue linki eklenir.

## 5. Kod ve deney kuralları

- Deney sonucu elle yazılmaz. Sonuçlar betikle üretilir, makaledeki tablolar o çıktıdan alınır.
- Test seti eğitimde kullanılmaz (Y-01). Fine-tune ya da ayar yaparken test bölümüne dokunma.
- Yeni bir ölçüm eklerken ölçüm koşullarını da yaz: donanım, sürüm, tur sayısı, seed.
- Okunur kod yaz: kısa fonksiyonlar, anlamlı adlar, uzun satırları böl, "neden" yorumu ekle.
