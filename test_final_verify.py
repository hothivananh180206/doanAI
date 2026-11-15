from ai.detector import detect_waste

img_path = r"C:\Users\hothi\.gemini\antigravity\brain\6fa6d8f8-5a52-4ab5-9f29-f1d5f3566c1f\.user_uploaded\media_1794763276870_14520d51.jpg"
with open(img_path, "rb") as f:
    b = f.read()

rgb, dets, ann, cls = detect_waste(b, 0.25)
print("Số lượng phát hiện:", len(dets))
for d in dets:
    print(f" -> Vật thể: {d['class_name']}")
    print(f" -> Nhóm rác: {d['category']}")
    print(f" -> Độ tin cậy: {d['confidence']:.1%}")
    print(f" -> Hướng dẫn: {d['instructions']}")
