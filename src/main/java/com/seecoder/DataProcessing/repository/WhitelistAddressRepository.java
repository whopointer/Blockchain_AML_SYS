package com.seecoder.DataProcessing.repository;

import com.seecoder.DataProcessing.po.WhitelistAddress;
import org.springframework.data.jpa.repository.JpaRepository;

public interface WhitelistAddressRepository extends JpaRepository<WhitelistAddress, Long> {
    boolean existsByAddress(String address);
}