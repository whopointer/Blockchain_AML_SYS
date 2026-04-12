/**
 * ETH值处理工具函数
 * 后端返回的ETH值为Wei单位，需要转换为显示格式
 */

/**
 * 将Wei转换为ETH
 * @param weiValue - Wei值（字符串或数字）
 * @returns ETH值字符串，保留适当的小数位
 */
export function weiToEth(weiValue: string | number | bigint): string {
  // 将输入转换为bigint以确保精确计算
  let weiBigInt: bigint;

  if (typeof weiValue === "bigint") {
    weiBigInt = weiValue;
  } else if (typeof weiValue === "string") {
    // 移除可能存在的小数点并转换为BigInt
    const cleanedValue = weiValue.split(".")[0];
    weiBigInt = BigInt(cleanedValue);
  } else if (typeof weiValue === "number") {
    // 检查数字是否可以安全转换为BigInt
    if (!Number.isInteger(weiValue) || !Number.isSafeInteger(weiValue)) {
      // 转换为整数字符串以避免小数问题
      const intValue = Math.floor(weiValue).toString();
      weiBigInt = BigInt(intValue);
    } else {
      weiBigInt = BigInt(weiValue);
    }
  } else {
    throw new Error("无效的wei值输入类型");
  }

  // 1 ETH = 10^18 Wei
  const divisor = BigInt(10 ** 18);

  // 计算整数部分
  const ethPart = weiBigInt / divisor;
  // 计算小数部分（Wei余数）
  const remainder = weiBigInt % divisor;

  // 转换为字符串，保持精度
  let result = ethPart.toString();

  if (remainder > 0) {
    // 将余数转换为ETH的小数部分
    let remainderStr = remainder.toString().padStart(18, "0");
    // 移除末尾的零
    remainderStr = remainderStr.replace(/0+$/, "");

    if (remainderStr.length > 0) {
      result += "." + remainderStr;
    }
  }

  return result;
}

/**
 * 格式化ETH值用于显示
 * @param weiValue - Wei值
 * @param decimals - 显示的小数位数（默认：6）
 * @returns 格式化后的ETH值字符串
 */
export function formatEthValue(
  weiValue: string | number | bigint,
  decimals: number = 6,
): string {
  const ethValue = weiToEth(weiValue);

  // 转换为数字并按指定小数位格式化
  const numValue = parseFloat(ethValue);
  if (isNaN(numValue)) {
    return "0";
  }

  // 处理非常小的值
  if (numValue === 0) {
    return "0";
  }

  // 使用toFixed限制小数位，然后移除末尾的零
  const fixedValue = numValue.toFixed(decimals);
  return parseFloat(fixedValue).toString();
}

/**
 * 将ETH转换为Wei
 * @param ethValue - ETH值
 * @returns Wei值（bigint）
 */
export function ethToWei(ethValue: number | string): bigint {
  const ethNum = typeof ethValue === "string" ? parseFloat(ethValue) : ethValue;
  return BigInt(Math.round(ethNum * 10 ** 18));
}
