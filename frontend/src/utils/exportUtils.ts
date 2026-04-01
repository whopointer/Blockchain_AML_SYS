import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import { GraphSnapshot } from "../components/CaseDetails/types";
import { NodeItem, LinkItem } from "../components/GraphCommon/types";
import dayjs from "dayjs";

// 使用 jspdf-autotable 插件来更好地处理表格
// 注意：需要在项目中安装 jspdf-autotable: npm install jspdf-autotable

/**
 * 将图谱数据转换为CSV格式
 */
export const convertGraphToCSV = (
  nodes: NodeItem[],
  links: LinkItem[],
  snapshot: GraphSnapshot,
): string => {
  // 节点CSV
  const nodeHeaders = ["节点ID", "地址", "标签", "风险等级", "金额", "类型"];
  const nodeRows = nodes.map((node) => [
    node.id,
    node.addr || "",
    node.label || "",
    node.malicious === 1 ? "高风险" : node.malicious === 0 ? "正常" : "未知",
    node.value?.toString() || "0",
    node.image ? "标记地址" : "普通地址",
  ]);

  // 边CSV
  const linkHeaders = ["交易哈希", "来源地址", "目标地址", "金额", "时间"];
  const linkRows = links.map((link) => [
    (link.tx_hash_list || []).join(", "),
    link.from,
    link.to,
    link.val?.toString() || "0",
    link.tx_time || "",
  ]);

  const csvContent = [
    "=".repeat(50),
    "案件信息",
    "=".repeat(50),
    `案件标题,${snapshot.title}`,
    `案件描述,${snapshot.description || ""}`,
    `风险等级,${snapshot.riskLevel}`,
    `创建时间,${dayjs(snapshot.createTime).format("YYYY-MM-DD HH:mm:ss")}`,
    `中心地址,${snapshot.centerAddress || ""}`,
    `节点数量,${snapshot.nodeCount}`,
    `边数量,${snapshot.linkCount}`,
    `标签,${snapshot.tags.join(", ")}`,
    "",
    "=".repeat(50),
    "节点数据",
    "=".repeat(50),
    nodeHeaders.join(","),
    ...nodeRows.map((row) => row.join(",")),
    "",
    "=".repeat(50),
    "交易数据",
    "=".repeat(50),
    linkHeaders.join(","),
    ...linkRows.map((row) => row.join(",")),
  ].join("\n");

  return csvContent;
};

/**
 * 下载CSV文件
 */
export const downloadCSV = (content: string, filename: string) => {
  const blob = new Blob(["\ufeff" + content], {
    type: "text/csv;charset=utf-8;",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(link.href);
};

/**
 * 导出图谱为PNG图片
 */
export const exportGraphToPNG = async (
  elementId: string,
  filename: string,
): Promise<boolean> => {
  const element = document.getElementById(elementId);
  if (!element) {
    console.error("Element not found:", elementId);
    return false;
  }

  try {
    const canvas = await html2canvas(element, {
      backgroundColor: "#ffffff",
      scale: 2,
      useCORS: true,
      allowTaint: true,
    });

    const link = document.createElement("a");
    link.download = filename;
    link.href = canvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    return true;
  } catch (error) {
    console.error("导出PNG失败:", error);
    return false;
  }
};

/**
 * 导出完整 SVG 内容为 PNG
 * 计算所有节点的边界框，调整 viewBox 以包含全部内容
 */
export const exportFullGraphToPNG = async (
  svgElement: SVGSVGElement | null,
  filename: string,
): Promise<boolean> => {
  if (!svgElement) {
    console.error("SVG element not found");
    return false;
  }

  try {
    // 获取原始 g 元素和其 transform
    const originalGElement = svgElement.querySelector("g");
    const originalTransform = originalGElement?.getAttribute("transform");

    if (!originalGElement) {
      console.error("SVG group element not found");
      return false;
    }

    // 临时移除 transform 以获取正确的边界
    originalGElement.removeAttribute("transform");
    const groupBBox = originalGElement.getBBox();

    // 恢复原始 transform
    if (originalTransform) {
      originalGElement.setAttribute("transform", originalTransform);
    }

    if (!groupBBox || groupBBox.width === 0 || groupBBox.height === 0) {
      console.error("Could not calculate bounds from SVG group");
      return false;
    }

    const padding = 10;
    const minX = groupBBox.x - padding;
    const minY = groupBBox.y - padding;
    const width = groupBBox.width + padding * 2;
    const height = groupBBox.height + padding * 2;

    // 克隆 SVG
    const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

    // 获取内部的 g 元素并移除 transform
    const gElement = clonedSvg.querySelector("g");
    if (gElement) {
      gElement.removeAttribute("transform");
      gElement.setAttribute("transform", `translate(${-minX},${-minY})`);
    }

    clonedSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    clonedSvg.setAttribute("width", `${width}`);
    clonedSvg.setAttribute("height", `${height}`);

    // 移除边框和圆角，设置白色背景
    clonedSvg.style.backgroundColor = "white";
    clonedSvg.style.border = "none";
    clonedSvg.style.borderRadius = "0";

    // 将克隆的 SVG 转换为字符串
    const svgData = new XMLSerializer().serializeToString(clonedSvg);
    const svgBlob = new Blob([svgData], {
      type: "image/svg+xml;charset=utf-8",
    });
    const url = URL.createObjectURL(svgBlob);

    // 创建 Image 对象加载 SVG
    const img = new Image();

    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });

    // 创建 canvas 绘制完整图像
    const canvas = document.createElement("canvas");
    const scale = 2;
    canvas.width = width * scale;
    canvas.height = height * scale;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to get canvas context");
    }

    // 填充白色背景
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 绘制 SVG
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // 下载
    const link = document.createElement("a");
    link.download = filename;
    link.href = canvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(url);
    return true;
  } catch (error) {
    console.error("导出完整图谱PNG失败:", error);
    return false;
  }
};

/**
 * 获取风险等级标签
 */
const getRiskLevelLabel = (riskLevel: string): string => {
  switch (riskLevel) {
    case "HIGH":
      return "高风险";
    case "MEDIUM":
      return "中风险";
    case "LOW":
      return "低风险";
    default:
      return "未知";
  }
};

/**
 * 获取风险等级颜色
 */
const getRiskLevelColor = (riskLevel: string): string => {
  switch (riskLevel) {
    case "HIGH":
      return "#ff4d4f";
    case "MEDIUM":
      return "#faad14";
    case "LOW":
      return "#52c41a";
    default:
      return "#1890ff";
  }
};

/**
 * 生成PDF报告 - 使用 html2canvas 将整个内容区域转为图片，避免中文乱码
 */
export const generatePDFReport = async (
  snapshot: GraphSnapshot,
  nodes: NodeItem[],
  links: LinkItem[],
  graphElementId?: string,
): Promise<boolean> => {
  try {
    // 创建一个临时的 DOM 元素用于生成 PDF 内容
    const tempDiv = document.createElement("div");
    tempDiv.style.cssText = `
      position: fixed;
      left: -9999px;
      top: 0;
      width: 794px;
      background: white;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      font-size: 12px;
      line-height: 1.5;
      color: #333;
    `;

    // 构建 PDF 内容 HTML
    const riskColor = getRiskLevelColor(snapshot.riskLevel);
    const riskLabel = getRiskLevelLabel(snapshot.riskLevel);

    // 获取筛选条件信息
    const filterConfig = snapshot.filterConfig;
    let filterInfoHTML = "";
    if (filterConfig) {
      filterInfoHTML = `
        <div style="margin-bottom: 20px;">
          <h3 style="font-size: 16px; color: #333; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid #e8e8e8;">筛选条件</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; width: 120px; font-weight: 500;">数据筛选</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${filterConfig.txType === "all" ? "全部交易" : filterConfig.txType === "inflow" ? "流入交易" : "流出交易"}</td>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; width: 120px; font-weight: 500;">地址筛选</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${filterConfig.addrType === "all" ? "全部地址" : filterConfig.addrType === "tagged" ? "标记地址" : filterConfig.addrType === "malicious" ? "风险地址" : "普通地址"}</td>
            </tr>
            ${
              filterConfig.minAmount !== undefined ||
              filterConfig.maxAmount !== undefined
                ? `
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">金额筛选</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;" colspan="3">${filterConfig.minAmount !== undefined ? filterConfig.minAmount : "最小值"} - ${filterConfig.maxAmount !== undefined ? filterConfig.maxAmount : "最大值"}</td>
            </tr>
            `
                : ""
            }
            ${
              filterConfig.startDate || filterConfig.endDate
                ? `
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">时间筛选</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;" colspan="3">${filterConfig.startDate ? dayjs(filterConfig.startDate).format("YYYY-MM-DD HH:mm:ss") : "最早"} ~ ${filterConfig.endDate ? dayjs(filterConfig.endDate).format("YYYY-MM-DD HH:mm:ss") : "最近"}</td>
            </tr>
            `
                : ""
            }
          </table>
        </div>
      `;
    }

    // 获取图谱截图 - 使用 SVG 完整内容
    let graphImageData: string | null = null;
    if (graphElementId) {
      const graphElement = document.getElementById(graphElementId);
      if (graphElement) {
        try {
          // 等待图谱渲染完成
          await new Promise((resolve) => setTimeout(resolve, 500));

          // 查找 SVG 元素
          const svgElement = graphElement.querySelector("svg") as SVGSVGElement;
          if (svgElement) {
            // 获取原始 g 元素和其 transform
            const originalGElement = svgElement.querySelector("g");
            const originalTransform =
              originalGElement?.getAttribute("transform");

            // 临时移除 transform 以获取正确的边界
            if (originalGElement) {
              originalGElement.removeAttribute("transform");
            }

            // 计算图谱边界
            const groupBBox = originalGElement?.getBBox();

            // 恢复原始 transform
            if (originalGElement && originalTransform) {
              originalGElement.setAttribute("transform", originalTransform);
            }

            if (groupBBox && groupBBox.width > 0 && groupBBox.height > 0) {
              const padding = 10;
              const minX = groupBBox.x - padding;
              const minY = groupBBox.y - padding;
              const svgWidth = groupBBox.width + padding * 2;
              const svgHeight = groupBBox.height + padding * 2;

              // 克隆 SVG
              const clonedSvg = svgElement.cloneNode(true) as SVGSVGElement;

              // 获取内部的 g 元素并确保没有原始 transform，同时将内容平移到 (0,0)
              const gElement = clonedSvg.querySelector("g");
              if (gElement) {
                gElement.removeAttribute("transform");
                gElement.setAttribute(
                  "transform",
                  `translate(${-minX},${-minY})`,
                );
              }

              // 使用 origin-based viewBox，裁剪掉左上方冗余空白
              clonedSvg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
              clonedSvg.setAttribute("width", `${svgWidth}`);
              clonedSvg.setAttribute("height", `${svgHeight}`);
              clonedSvg.style.backgroundColor = "white";
              clonedSvg.style.border = "none";
              clonedSvg.style.borderRadius = "0";

              // 将 SVG 转换为图片
              const svgData = new XMLSerializer().serializeToString(clonedSvg);
              const svgBlob = new Blob([svgData], {
                type: "image/svg+xml;charset=utf-8",
              });
              const url = URL.createObjectURL(svgBlob);

              const img = new Image();
              await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = url;
              });

              // 创建 canvas
              const canvas = document.createElement("canvas");
              const scale = 2;
              canvas.width = svgWidth * scale;
              canvas.height = svgHeight * scale;

              const ctx = canvas.getContext("2d");
              if (ctx) {
                ctx.fillStyle = "white";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                graphImageData = canvas.toDataURL("image/png");
              }

              URL.revokeObjectURL(url);
            }
          }
        } catch (error) {
          console.error("生成图谱截图失败:", error);
        }
      }
    }

    tempDiv.innerHTML = `
      <div style="padding: 30px;">
        <!-- 标题 -->
        <div style="text-align: center; margin-bottom: 24px;">
          <h1 style="font-size: 22px; color: #667eea; margin: 0 0 8px 0; font-weight: 600;">区块链AML案件报告 - ${snapshot.title || "未命名快照"}</h1>
          <div style="width: 60px; height: 3px; background: #667eea; margin: 0 auto;"></div>
        </div>

        <!-- 案件信息 -->
        <div style="margin-bottom: 20px;">
          <h3 style="font-size: 16px; color: #333; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid #e8e8e8;">案件信息</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; width: 120px; font-weight: 500;">快照 ID</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${snapshot.id}</td>
            </tr>
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">标题</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8; font-weight: 500;">${snapshot.title}</td>
            </tr>
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">描述</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${snapshot.description || "无"}</td>
            </tr>
            ${
              snapshot.centerAddress
                ? `
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">中心地址</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8; font-family: monospace; font-size: 11px; word-break: break-all;">${snapshot.centerAddress}</td>
            </tr>
            `
                : `
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">起始地址</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8; font-family: monospace; font-size: 11px; word-break: break-all;">${snapshot.fromAddress || ""}</td>
            </tr>
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">目标地址</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8; font-family: monospace; font-size: 11px; word-break: break-all;">${snapshot.toAddress || ""}</td>
            </tr>
            `
            }
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">创建时间</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${dayjs(snapshot.createTime).format("YYYY-MM-DD HH:mm:ss")}</td>
            </tr>
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">风险等级</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">
                <span style="display: inline-block; padding: 2px 8px; background: ${riskColor}20; color: ${riskColor}; border-radius: 4px; font-size: 12px;">${riskLabel}</span>
              </td>
            </tr>
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">标签</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">
                ${
                  snapshot.tags.length > 0
                    ? snapshot.tags
                        .map(
                          (tag) =>
                            `<span style="display: inline-block; padding: 2px 8px; background: #e6f7ff; color: #1890ff; border-radius: 4px; font-size: 12px; margin-right: 4px;">${tag}</span>`,
                        )
                        .join("")
                    : "无"
                }
              </td>
            </tr>
          </table>
        </div>

        <!-- 统计信息 -->
        <div style="margin-bottom: 20px;">
          <h3 style="font-size: 16px; color: #333; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid #e8e8e8;">统计信息</h3>
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; width: 120px; font-weight: 500;">总节点数</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${nodes.length}</td>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; width: 120px; font-weight: 500;">总交易数</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${links.length}</td>
            </tr>
            <tr>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">高风险节点</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${nodes.filter((n) => n.malicious === 1).length}</td>
              <td style="padding: 8px; background: #fafafa; border: 1px solid #e8e8e8; font-weight: 500;">标记节点</td>
              <td style="padding: 8px; border: 1px solid #e8e8e8;">${nodes.filter((n) => n.image).length}</td>
            </tr>
          </table>
        </div>

        ${filterInfoHTML}

        <!-- 页脚 -->
        <div style="margin-top: 30px; padding-top: 12px; border-top: 1px solid #e8e8e8; text-align: center; color: #999; font-size: 10px;">
          区块链AML系统 - 生成时间: ${dayjs().format("YYYY-MM-DD HH:mm:ss")}
        </div>
      </div>
    `;

    document.body.appendChild(tempDiv);

    // 使用 html2canvas 将 DOM 转为图片
    const canvas = await html2canvas(tempDiv, {
      backgroundColor: "#ffffff",
      scale: 2,
      useCORS: true,
      allowTaint: true,
      logging: false,
    });

    // 创建 PDF
    const pdf = new jsPDF("p", "mm", "a4");
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 0;

    const imgData = canvas.toDataURL("image/png");
    const imgWidth = pageWidth - margin * 2;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    // 计算需要多少页
    let heightLeft = imgHeight;
    let position = margin;
    let pageCount = 0;

    // 添加第一页
    pdf.addImage(imgData, "PNG", margin, position, imgWidth, imgHeight);
    heightLeft -= pageHeight;

    // 如果内容超出第一页，添加更多页
    while (heightLeft > 0) {
      pageCount++;
      position = -pageHeight * pageCount + margin;
      pdf.addPage();
      pdf.addImage(imgData, "PNG", margin, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;
    }

    // 如果有图谱图片，单独添加一页显示完整图谱
    if (graphImageData) {
      pdf.addPage();

      // 计算图谱图片尺寸，保持比例并适应页面
      const maxImgWidth = pageWidth - margin * 2 - 10;
      const maxImgHeight = pageHeight - margin * 2 - 20;

      // 创建临时图片对象获取原始尺寸
      const tempImg = new Image();
      const imgData = graphImageData; // 保存到局部变量避免 null 检查问题
      await new Promise((resolve) => {
        tempImg.onload = resolve;
        tempImg.src = imgData;
      });

      const imgOriginalWidth = tempImg.width || 800;
      const imgOriginalHeight = tempImg.height || 600;

      // 计算缩放比例，保持宽高比
      const widthRatio = maxImgWidth / imgOriginalWidth;
      const heightRatio = maxImgHeight / imgOriginalHeight;
      const scale = Math.min(widthRatio, heightRatio);

      let imgDisplayWidth = imgOriginalWidth * scale;
      let imgDisplayHeight = imgOriginalHeight * scale;

      // 居中显示
      const imgX = (pageWidth - imgDisplayWidth) / 2;
      const imgY = (pageHeight - imgDisplayHeight) / 2;

      pdf.addImage(
        imgData,
        "PNG",
        imgX,
        imgY,
        imgDisplayWidth,
        imgDisplayHeight,
      );
    }

    // 保存 PDF
    pdf.save(`${snapshot.title}-案件报告.pdf`);

    // 清理临时元素
    document.body.removeChild(tempDiv);

    return true;
  } catch (error) {
    console.error("生成PDF报告失败:", error);
    return false;
  }
};

/**
 * 导出完整案件包（PDF + CSV + PNG）
 */
export const exportCasePackage = async (
  snapshot: GraphSnapshot,
  nodes: NodeItem[],
  links: LinkItem[],
  graphElementId?: string,
): Promise<{ success: boolean; message: string }> => {
  const results: string[] = [];

  try {
    // 1. 导出PDF报告
    const pdfSuccess = await generatePDFReport(
      snapshot,
      nodes,
      links,
      graphElementId,
    );
    if (pdfSuccess) {
      results.push("PDF报告");
    }

    // 2. 导出CSV数据
    const csvContent = convertGraphToCSV(nodes, links, snapshot);
    downloadCSV(csvContent, `${snapshot.title}-数据.csv`);
    results.push("CSV数据");

    // 3. 导出PNG图片（如果有图谱元素）- 导出完整 SVG 内容
    if (graphElementId) {
      const container = document.getElementById(graphElementId);
      const svgElement = container?.querySelector(
        "svg",
      ) as SVGSVGElement | null;

      const pngSuccess = await exportFullGraphToPNG(
        svgElement,
        `${snapshot.title}-图谱.png`,
      );
      if (pngSuccess) {
        results.push("PNG图谱");
      }
    }

    return {
      success: true,
      message: `成功导出: ${results.join(", ")}`,
    };
  } catch (error) {
    console.error("导出案件包失败:", error);
    return {
      success: false,
      message: "导出失败: " + (error as Error).message,
    };
  }
};
