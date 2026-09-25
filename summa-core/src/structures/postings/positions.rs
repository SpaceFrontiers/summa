//! Packed element ordinals and token positions for indexed text.

/// Maximum token position within an element (20 bits = 1,048,575)
pub const MAX_TOKEN_POSITION: u32 = (1 << 20) - 1;

/// Maximum element ordinal (12 bits = 4095)
pub const MAX_ELEMENT_ORDINAL: u32 = (1 << 12) - 1;

/// Encode element ordinal and token position into a single u32
#[inline]
pub fn encode_position(element_ordinal: u32, token_position: u32) -> u32 {
    debug_assert!(
        element_ordinal <= MAX_ELEMENT_ORDINAL,
        "Element ordinal {} exceeds maximum {}",
        element_ordinal,
        MAX_ELEMENT_ORDINAL
    );
    debug_assert!(
        token_position <= MAX_TOKEN_POSITION,
        "Token position {} exceeds maximum {}",
        token_position,
        MAX_TOKEN_POSITION
    );
    (element_ordinal << 20) | (token_position & MAX_TOKEN_POSITION)
}

/// Decode element ordinal from encoded position
#[inline]
pub fn decode_element_ordinal(position: u32) -> u32 {
    position >> 20
}

/// Decode token position from encoded position
#[inline]
pub fn decode_token_position(position: u32) -> u32 {
    position & MAX_TOKEN_POSITION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_encoding() {
        // Element 0, position 5
        let pos = encode_position(0, 5);
        assert_eq!(decode_element_ordinal(pos), 0);
        assert_eq!(decode_token_position(pos), 5);

        // Element 3, position 100
        let pos = encode_position(3, 100);
        assert_eq!(decode_element_ordinal(pos), 3);
        assert_eq!(decode_token_position(pos), 100);

        // Max values
        let pos = encode_position(MAX_ELEMENT_ORDINAL, MAX_TOKEN_POSITION);
        assert_eq!(decode_element_ordinal(pos), MAX_ELEMENT_ORDINAL);
        assert_eq!(decode_token_position(pos), MAX_TOKEN_POSITION);
    }
}
