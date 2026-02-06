package com.seecoder.DataProcessing.serviceImpl.graph;

import com.seecoder.DataProcessing.repository.graph.AddressNodeRepository;
import com.seecoder.DataProcessing.repository.graph.TransactionNodeRepository;
import com.seecoder.DataProcessing.repository.graph.TransferRelationRepository;
import lombok.extern.slf4j.Slf4j;
import org.neo4j.ogm.session.Session;
import org.neo4j.ogm.session.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import java.math.BigDecimal;
import java.util.*;

@Slf4j
public abstract class AbstractGraphService {

    @Autowired
    protected SessionFactory sessionFactory;

    @Autowired
    protected AddressNodeRepository addressNodeRepository;

    @Autowired
    protected TransactionNodeRepository transactionNodeRepository;

    @Autowired
    protected TransferRelationRepository transferRelationRepository;

    // 获取Session
    protected Session getSession() {
        return sessionFactory.openSession();
    }

    // 清理Session
    protected void closeSession(Session session) {
        if (session != null) {
            session.clear();
        }
    }

    protected Double convertBigDecimal(BigDecimal value) {
        return value != null ? value.doubleValue() : 0.0;
    }

    // 添加通用的安全转换方法
    protected <T> List<T> safeConvertToList(Object value, Class<T> targetType) {
        List<T> result = new ArrayList<>();

        if (value == null) {
            return result;
        }

        try {
            if (value instanceof List) {
                List<?> list = (List<?>) value;
                for (Object item : list) {
                    if (item != null) {
                        T convertedItem = convertValue(item, targetType);
                        if (convertedItem != null) {
                            result.add(convertedItem);
                        }
                    } else {
                        if (targetType == String.class) {
                            result.add((T) "");
                        } else if (targetType == Integer.class) {
                            result.add((T) Integer.valueOf(0));
                        } else if (targetType == Double.class) {
                            result.add((T) Double.valueOf(0.0));
                        } else {
                            result.add(null);
                        }
                    }
                }
                return result;
            } else if (value.getClass().isArray()) {
                if (targetType == String.class && value instanceof String[]) {
                    return (List<T>) Arrays.asList((String[]) value);
                } else if (targetType == Integer.class && value instanceof Integer[]) {
                    return (List<T>) Arrays.asList((Integer[]) value);
                } else if (targetType == Double.class && value instanceof Double[]) {
                    return (List<T>) Arrays.asList((Double[]) value);
                } else if (targetType == Double.class && value instanceof Float[]) {
                    // 处理Float数组转Double
                    Float[] floatArray = (Float[]) value;
                    for (Float f : floatArray) {
                        result.add((T) Double.valueOf(f));
                    }
                    return result;
                } else if (value instanceof Object[]) {
                    for (Object obj : (Object[]) value) {
                        result.add(convertValue(obj, targetType));
                    }
                    return result;
                }
            }

            // 其他情况尝试直接转换
            log.warn("无法直接转换的类型: {} to {}", value.getClass(), targetType);
        } catch (Exception e) {
            log.error("类型转换失败", e);
        }

        return result;
    }

    protected <T> T convertValue(Object value, Class<T> targetType) {
        if (value == null) {
            if (targetType == Double.class) return (T) Double.valueOf(0.0);
            if (targetType == Integer.class) return (T) Integer.valueOf(0);
            if (targetType == String.class) return (T) "";
            return null;
        }

        try {
            // 处理Neo4j特有的节点和关系对象
            if (value.getClass().getSimpleName().contains("NodeAdapter") || 
                value.getClass().getSimpleName().contains("EntityAdapter")) {
                // Try to extract properties from Neo4j node-like objects
                if (targetType == String.class) {
                    return (T) value.toString();
                } else if (targetType == Double.class) {
                    return (T) Double.valueOf(0.0);
                } else if (targetType == Integer.class) {
                    return (T) Integer.valueOf(0);
                }
            }

            if (targetType == Double.class) {
                if (value instanceof Number) {
                    return (T) Double.valueOf(((Number) value).doubleValue());
                } else if (value instanceof String) {
                    return (T) Double.valueOf(Double.parseDouble((String) value));
                }
            } else if (targetType == Integer.class) {
                if (value instanceof Number) {
                    return (T) Integer.valueOf(((Number) value).intValue());
                } else if (value instanceof String) {
                    return (T) Integer.valueOf(Integer.parseInt((String) value));
                }
            } else if (targetType == String.class) {
                return (T) value.toString();
            }
        } catch (Exception e) {
            log.error("值转换失败: {} to {}", value, targetType, e);
        }

        // 默认值
        if (targetType == Double.class) return (T) Double.valueOf(0.0);
        if (targetType == Integer.class) return (T) Integer.valueOf(0);
        if (targetType == String.class) return (T) "";

        return null;
    }

    // 添加辅助方法处理数组转换
    protected List<String> convertToStringList(Object value) {
        List<String> result = new ArrayList<>();
        if (value == null) {
            return result;
        }

        if (value instanceof List) {
            List<?> list = (List<?>) value;
            for (Object obj : list) {
                if (obj != null) {
                    result.add(obj.toString());
                } else {
                    result.add("");
                }
            }
            return result;
        } else if (value.getClass().isArray()) {
            if (value instanceof String[]) {
                return Arrays.asList((String[]) value);
            } else if (value instanceof Object[]) {
                for (Object obj : (Object[]) value) {
                    result.add(obj != null ? obj.toString() : "");
                }
            }
        }
        return result;
    }

    protected List<Integer> convertToIntegerList(Object value) {
        List<Integer> result = new ArrayList<>();
        if (value == null) {
            return result;
        }

        if (value instanceof List) {
            List<?> list = (List<?>) value;
            for (Object obj : list) {
                if (obj instanceof Number) {
                    result.add(((Number) obj).intValue());
                } else if (obj != null) {
                    try {
                        result.add(Integer.parseInt(obj.toString()));
                    } catch (NumberFormatException e) {
                        result.add(0);
                    }
                } else {
                    result.add(0);
                }
            }
            return result;
        } else if (value.getClass().isArray()) {
            if (value instanceof Integer[]) {
                return Arrays.asList((Integer[]) value);
            } else if (value instanceof int[]) {
                for (int num : (int[]) value) {
                    result.add(num);
                }
            } else if (value instanceof Number[]) {
                for (Number num : (Number[]) value) {
                    result.add(num != null ? num.intValue() : 0);
                }
            }
        }
        return result;
    }

    protected List<Double> convertToDoubleList(Object value) {
        List<Double> result = new ArrayList<>();
        if (value == null) {
            return result;
        }

        if (value instanceof List) {
            List<?> list = (List<?>) value;
            for (Object obj : list) {
                if (obj instanceof Number) {
                    result.add(((Number) obj).doubleValue());
                } else if (obj != null) {
                    try {
                        result.add(Double.parseDouble(obj.toString()));
                    } catch (NumberFormatException e) {
                        result.add(0.0);
                    }
                } else {
                    result.add(0.0);
                }
            }
            return result;
        } else if (value.getClass().isArray()) {
            if (value instanceof Double[]) {
                return Arrays.asList((Double[]) value);
            } else if (value instanceof double[]) {
                for (double num : (double[]) value) {
                    result.add(num);
                }
            } else if (value instanceof Number[]) {
                for (Number num : (Number[]) value) {
                    result.add(num != null ? num.doubleValue() : 0.0);
                }
            }
        }
        return result;
    }
}