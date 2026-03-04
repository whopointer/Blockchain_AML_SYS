/**
 * Utility functions for handling ETH values
 * ETH values come in Wei from the backend and need to be converted for display
 */

/**
 * Convert Wei to ETH
 * @param weiValue - Value in Wei (as string or number)
 * @returns Value in ETH as string with appropriate decimal places
 */
export function weiToEth(weiValue: string | number | bigint): string {
  // Convert input to bigint for precise calculation
  let weiBigInt: bigint;

  if (typeof weiValue === "bigint") {
    weiBigInt = weiValue;
  } else if (typeof weiValue === "string") {
    // Remove decimal point if present and convert to BigInt
    const cleanedValue = weiValue.split(".")[0];
    weiBigInt = BigInt(cleanedValue);
  } else if (typeof weiValue === "number") {
    // Check if the number is safe to convert to BigInt
    if (!Number.isInteger(weiValue) || !Number.isSafeInteger(weiValue)) {
      // Convert to integer string to avoid decimal issues
      const intValue = Math.floor(weiValue).toString();
      weiBigInt = BigInt(intValue);
    } else {
      weiBigInt = BigInt(weiValue);
    }
  } else {
    throw new Error("Invalid input type for wei value");
  }

  // 1 ETH = 10^18 Wei
  const divisor = BigInt(10 ** 18);

  // Calculate whole ETH part
  const ethPart = weiBigInt / divisor;
  // Calculate fractional part (remainder in Wei)
  const remainder = weiBigInt % divisor;

  // Convert to number for decimal calculation, but keep precision
  let result = ethPart.toString();

  if (remainder > 0) {
    // Convert remainder to decimal part of ETH
    let remainderStr = remainder.toString().padStart(18, "0");
    // Remove trailing zeros
    remainderStr = remainderStr.replace(/0+$/, "");

    if (remainderStr.length > 0) {
      result += "." + remainderStr;
    }
  }

  return result;
}

/**
 * Format ETH value for display
 * @param weiValue - Value in Wei
 * @param decimals - Number of decimal places to show (default: 6)
 * @returns Formatted ETH value as string
 */
export function formatEthValue(
  weiValue: string | number | bigint,
  decimals: number = 6,
): string {
  const ethValue = weiToEth(weiValue);

  // Convert to number and format with specified decimals
  const numValue = parseFloat(ethValue);
  if (isNaN(numValue)) {
    return "0";
  }

  // Handle very small values
  if (numValue === 0) {
    return "0";
  }

  // Use toFixed to limit decimal places, but remove trailing zeros
  const fixedValue = numValue.toFixed(decimals);
  return parseFloat(fixedValue).toString();
}

/**
 * Convert ETH to Wei
 * @param ethValue - Value in ETH
 * @returns Value in Wei as bigint
 */
export function ethToWei(ethValue: number | string): bigint {
  const ethNum = typeof ethValue === "string" ? parseFloat(ethValue) : ethValue;
  return BigInt(Math.round(ethNum * 10 ** 18));
}
