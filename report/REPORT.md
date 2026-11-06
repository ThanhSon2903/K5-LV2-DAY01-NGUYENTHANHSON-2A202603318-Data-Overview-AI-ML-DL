## Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 3.13.6 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> File này là bản **đã điền** để dùng thử upload Colab / nộp **report/**. Số liệu record dưới đây là ví dụ điển hình cho sample **traffic** / **kitchen**; nếu JSON máy bạn ra khác thì thay đúng số từ file output của bạn. Không ghi họ tên, MSSV hay dữ liệu cá nhân.
### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** trong ảnh. Vì classification single-label của checkpoint này gán một lớp cho toàn ảnh, không tạo một box/object riêng trong JSON.
3. Class list đi kèm taxonomy/checkpoint (ImageNet-1K với yolo11n-cls.pt), không phải model tự nghĩ ra tên lớp.
4. Cần cả ba vì: class_id ổn định cho máy; class_name dễ đọc cho người; taxonomy_name cho biết đang dùng bộ lớp nào (tránh nhầm khi đổi dataset/checkpoint).
5. Guideline cần nói rõ chọn lớp theo tiêu chí nào khi ảnh nhiều chủ đề (ví dụ: chủ thể lớn nhất / chủ đề chính của cảnh / lớp xuất hiện rõ nhất), để mọi annotator chọn thống nhất một lớp.
6. Vì score chỉ là độ tin của model trên prediction; nó không được người xác nhận theo guideline, và top-1 ImageNet trên ảnh COCO có thể trông “lạ” nhưng vẫn chỉ là prediction.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name` = bowl ; `score`= 0.701282 ; `bbox_xyxy`=[155.02,168.67,182.38,184.18] ; `bbox_width`=27.36 ; `bbox_height`= 15.51
2. Lớp `bowl`, score = 0.701282; box từ trên-trái khoảng pixel(155.02,168.67) đến góc dưới phải(182.38,184.18); rộng khoảng 27 px; cao khoảng 16 px.
3. Ví dụ so hai ngưỡng trên ảnh `kitchen`: threshold 0.20 → nhiều prediction hơn (vd. 18); threshold 0.60 → ít hơn (vd. 7). 
4. Reviewer phải xem thêm nhiều hộp hơn, kể cả hộp nhiễu/score thấp, nên khối lượng kiểm tra tăng.
5. Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. Trên ảnh `kitchen`, vật bị cắt mép bàn/đồ dùng gần biên ảnh: Chọn `guideline` nếu rule đã có (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì escalation hỏi Lab Coach, không tự bịa lớp.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-006 ; `class_name`= bowl ; `score`= 0.528201 ; số điểm polygon=`polygon_point_count`=84 ; 3–5 điểm đầu của `polygon_xy`= [
    [29.0,303.0],
    [28.0,304.0],
    [26.0,304.0],
    [25.0,305.0],
    [23.0,305.0]
]
2. Polygon bám theo biên thật của object (cong/mép), còn box chỉ là hình chữ nhật bao ngoài nên mất chi tiết hình dạng.
3. `instance_id` dùng để phân biệt từng object trong output lab (vd. hai `bowl` khác nhau là `kitchen-001` và `kitchen-002`). Nó không phải mã lớp (`class_id`) và không phải ID theo dõi theo thời gian (tracking).
4. Quy tắc biên mask: Mask phải đi sát biên nhìn thấy của instance; không tràn sang object bên cạnh; vùng bị che khuất thì chỉ vẽ phần còn nhìn thấy (trừ khi guideline nói khác).
5. Chỗ hai đồ dùng chạm nhau trên bàn `kitchen`: nếu guideline chưa nói rõ tách biên thế nào thì escalation; nếu guideline đã bảo “tách theo khe nhìn thấy giữa hai object” thì làm theo guideline.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Mỗi ảnh được gán duy nhất một nhãn đại diện cho nội dung chính, thuộc hệ thống phân cấp ImageNet-1K. | Mô hình có thể dự đoán Top-1 chưa sát với nội dung thực tế. Một số cảnh có nhiều đối tượng khiến việc xác định lớp đại diện dễ nhầm lẫn. | Quan sát toàn bộ ảnh, đối chiếu định nghĩa trong guideline và chọn một nhãn phù hợp nhất, không tự suy diễn theo điểm số của mô hình | Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn |
| Phát hiện vật thể |Mỗi đối tượng cần được gán một nhãn class và một bounding box theo tọa độ xyxy.|Hạ threshold ra thêm hộp nhiễu; object cắt mép  |Vẽ box theo rule, đủ object trong guideline  | Box có ôm sát không, thiếu/thừa object, ca che khuất |
| Instance segmentation |Mỗi instance có class và mask riêng theo từng pixel.  |Biên hai vật chạm nhau / vùng mờ  | Vẽ mask bám biên từng instance | Mask có tràn object khác không, đủ instance cùng lớp | 

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng ngay quá trình xử lý, không tự ý chỉnh sửa hay phát tán dữ liệu và báo cho Lab Coach hoặc giảng viên phụ trách để được xác minh, hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
