// com/seecoder/DataProcessing/util/DateUtil.java
package com.seecoder.DataProcessing.util;

import java.sql.Timestamp;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Date;

public class DateUtil {

    private static final DateTimeFormatter FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final SimpleDateFormat DATE_FORMAT =
            new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    public static Timestamp toTimestamp(Date date) {
        if (date == null) return null;
        return new Timestamp(date.getTime());
    }

    public static Date parseDate(String dateString) throws ParseException {
        return DATE_FORMAT.parse(dateString);
    }

    public static String formatDate(Date date) {
        if (date == null) return null;
        LocalDateTime localDateTime = date.toInstant()
                .atZone(ZoneId.systemDefault())
                .toLocalDateTime();
        return localDateTime.format(FORMATTER);
    }

    public static Date addDays(Date date, int days) {
        Instant instant = date.toInstant().plusSeconds(days * 24 * 60 * 60);
        return Date.from(instant);
    }
}