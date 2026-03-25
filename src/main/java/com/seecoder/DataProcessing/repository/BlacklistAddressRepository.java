package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.BlacklistAddress;
import org.springframework.data.jpa.repository.JpaRepository;

public interface BlacklistAddressRepository extends JpaRepository<BlacklistAddress, Long> {
    boolean existsByAddress(String address);
}