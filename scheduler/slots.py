def pad_slot_ids(slot_ids: list[int], batch_size: int, unused_slot: int):
    missing = batch_size - len(slot_ids)
    if missing < 0:
        raise ValueError(
            "more elements than batch size"
        )
    elif missing == 0:
        return slot_ids
    else:
        return slot_ids + [unused_slot] * missing
