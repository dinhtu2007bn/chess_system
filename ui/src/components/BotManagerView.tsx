import React, { useRef, useState } from "react";
import { ManagedBot } from "../types";

type Props = {
  bots: ManagedBot[];
  onRefresh: () => void;
  onSelectForBenchmark?: (botId: string) => void;
  onSelectForPlay?: (botId: string) => void;
};

export const BotManagerView: React.FC<Props> = ({
  bots,
  onRefresh,
  onSelectForBenchmark,
  onSelectForPlay,
}) => {
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [botName, setBotName] = useState("");
  const [botDesc, setBotDesc] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [fileCountText, setFileCountText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // In-place rename state
  const [editingBotId, setEditingBotId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [renaming, setRenaming] = useState(false);

  // Delete state
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      setSelectedFiles(null);
      setFileCountText("");
      return;
    }
    setSelectedFiles(files);
    let hasBotPy = false;
    for (let i = 0; i < files.length; i++) {
      const p = (files[i] as any).webkitRelativePath || files[i].name;
      if (p.endsWith("bot.py")) {
        hasBotPy = true;
        break;
      }
    }
    setFileCountText(
      `${files.length} tệp đã chọn ${hasBotPy ? "✓ (Đã tìm thấy bot.py)" : "⚠️ (Chưa thấy bot.py)"}`
    );
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!botName.trim()) {
      setUploadError("Vui lòng nhập tên hiển thị cho bot");
      return;
    }
    if (!selectedFiles || selectedFiles.length === 0) {
      setUploadError("Vui lòng chọn thư mục chứa mã nguồn bot (thư mục student)");
      return;
    }

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append("name", botName.trim());
    formData.append("description", botDesc.trim());

    const relativePaths: string[] = [];
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      const rel = (file as any).webkitRelativePath || file.name;
      relativePaths.push(rel);
      formData.append("files", file);
    }
    formData.append("relative_paths", JSON.stringify(relativePaths));

    try {
      const res = await fetch("/api/bots/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Không thể tải lên bot");
      }
      setShowUploadModal(false);
      setBotName("");
      setBotDesc("");
      setSelectedFiles(null);
      setFileCountText("");
      onRefresh();
    } catch (err: any) {
      setUploadError(err.message || "Lỗi khi tải lên bot");
    } finally {
      setUploading(false);
    }
  };

  const startRename = (bot: ManagedBot) => {
    setEditingBotId(bot.id);
    setEditName(bot.name);
  };

  const saveRename = async (botId: string) => {
    if (!editName.trim()) return;
    setRenaming(true);
    try {
      const res = await fetch(`/api/bots/${botId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: editName.trim() }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Lỗi khi đổi tên bot");
      }
      setEditingBotId(null);
      onRefresh();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setRenaming(false);
    }
  };

  const handleDeleteBot = async (botId: string, botName: string) => {
    if (!confirm(`Bạn có chắc chắn muốn xóa bot "${botName}" không? Thao tác này sẽ dọn dẹp toàn bộ file của bot trên hệ thống.`)) {
      return;
    }
    setDeletingId(botId);
    try {
      const res = await fetch(`/api/bots/${botId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Không thể xóa bot");
      }
      onRefresh();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="bot-manager-container">
      <div className="section-header">
        <div>
          <h2>Quản lý & Thêm Bot Cờ Vua Chess960</h2>
          <p className="section-desc">
            Nạp trực tiếp toàn bộ thư mục <code>student</code> từ máy tính, đổi tên phiên bản linh hoạt, và chạy độc lập trong tiến trình con.
          </p>
        </div>
        <button
          className="btn-primary"
          onClick={() => {
            setShowUploadModal(true);
            setUploadError(null);
          }}
        >
          + Thêm Bot Mới (Upload Folder)
        </button>
      </div>

      <div className="bots-grid">
        {bots.length === 0 ? (
          <div className="empty-state">
            <p>Chưa có bot nào trong hệ thống. Hãy bấm <strong>+ Thêm Bot Mới</strong> để nạp bot của bạn!</p>
          </div>
        ) : (
          bots.map((bot) => {
            const isEditing = editingBotId === bot.id;
            return (
              <div key={bot.id} className={`bot-card ${bot.is_default ? "default-bot" : ""}`}>
                <div className="bot-card-header">
                  {isEditing ? (
                    <div className="rename-form">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        placeholder="Nhập tên mới..."
                        autoFocus
                      />
                      <button
                        className="btn-sm btn-success"
                        disabled={renaming}
                        onClick={() => saveRename(bot.id)}
                      >
                        {renaming ? "Lưu..." : "Lưu"}
                      </button>
                      <button
                        className="btn-sm btn-secondary"
                        onClick={() => setEditingBotId(null)}
                      >
                        Hủy
                      </button>
                    </div>
                  ) : (
                    <div className="bot-title-row">
                      <h3 className="bot-name" title={bot.name}>
                        {bot.name}
                      </h3>
                      <button
                        className="btn-icon"
                        title="Đổi tên bot trực tiếp"
                        onClick={() => startRename(bot)}
                      >
                        ✏️
                      </button>
                    </div>
                  )}
                  <div className="bot-badge-group">
                    {bot.is_default && <span className="badge badge-default">Mặc định</span>}
                    {bot.is_valid ? (
                      <span className="badge badge-valid">✓ Hợp lệ</span>
                    ) : (
                      <span className="badge badge-invalid">✕ Lỗi file</span>
                    )}
                  </div>
                </div>

                <p className="bot-desc">{bot.description || "Không có mô tả chi tiết."}</p>

                <div className="bot-meta">
                  <div className="meta-item">
                    <span className="meta-label">Mã định danh:</span>
                    <code className="meta-code">{bot.id}</code>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Thư mục nạp:</span>
                    <span className="meta-path" title={bot.dir_path || ""}>
                      {bot.dir_name}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Thời gian nạp:</span>
                    <span>
                      {bot.created_at ? new Date(bot.created_at).toLocaleString("vi-VN") : "N/A"}
                    </span>
                  </div>
                </div>

                <div className="bot-card-actions">
                  {onSelectForPlay && (
                    <button
                      className="btn-outline btn-sm"
                      onClick={() => onSelectForPlay(bot.id)}
                    >
                      🎮 Đấu Thử Với Người
                    </button>
                  )}
                  {onSelectForBenchmark && (
                    <button
                      className="btn-primary btn-sm"
                      onClick={() => onSelectForBenchmark(bot.id)}
                    >
                      ⚡ Chạy Benchmark
                    </button>
                  )}
                  {!bot.is_default && (
                    <button
                      className="btn-danger btn-sm"
                      disabled={deletingId === bot.id}
                      onClick={() => handleDeleteBot(bot.id, bot.name)}
                    >
                      {deletingId === bot.id ? "Đang xóa..." : "Xóa"}
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {showUploadModal && (
        <div className="modal-backdrop">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Thêm Bot Mới Vào Hệ Thống</h3>
              <button
                className="btn-close"
                onClick={() => setShowUploadModal(false)}
              >
                ✕
              </button>
            </div>
            <form onSubmit={handleUploadSubmit}>
              <div className="modal-body">
                {uploadError && (
                  <div className="alert-box alert-danger">{uploadError}</div>
                )}
                <div className="form-group">
                  <label>Tên hiển thị bot (VD: Bot v2 - Smart Timer, Bot v3 - Deep Search):</label>
                  <input
                    type="text"
                    required
                    value={botName}
                    onChange={(e) => setBotName(e.target.value)}
                    placeholder="Nhập tên bot..."
                  />
                </div>
                <div className="form-group">
                  <label>Mô tả ngắn về phiên bản / thuật toán:</label>
                  <input
                    type="text"
                    value={botDesc}
                    onChange={(e) => setBotDesc(e.target.value)}
                    placeholder="Mô tả cải tiến thuật toán tìm kiếm, hàm lượng giá, bộ đếm giờ..."
                  />
                </div>
                <div className="form-group">
                  <label>Chọn cả thư mục <code>student</code> từ máy tính:</label>
                  <p className="input-hint">
                    Hệ thống sẽ nạp toàn bộ các file (<code>bot.py</code>, <code>search.py</code>, <code>evaluation.py</code>, <code>time_manager.py</code>...).
                  </p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    required
                    {...({ webkitdirectory: "", directory: "", multiple: true } as any)}
                    onChange={handleFileChange}
                    className="file-input"
                  />
                  {fileCountText && <p className="file-preview-text">{fileCountText}</p>}
                </div>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => setShowUploadModal(false)}
                >
                  Hủy Bỏ
                </button>
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={uploading}
                >
                  {uploading ? "Đang nạp & kiểm tra bot..." : "Nạp Bot & Lưu Trữ"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

