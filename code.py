def potential_guided_best_swap(initial_A: List[Set[Any]], V: Callable, w: Callable) -> List[Set[Any]]:
    """
    Algorithm 2: Potential-Guided Best-Swap Algorithm
    
    :param initial_A: The starting allocation (list of sets representing agent bundles)
    :param V: Valuation function V(i, bundle)
    :param w: Weight/Value function w(j, item) for finding the least-valued item
    :return: An EFX allocation
    """
    # Initialize allocation A = (A1, A2, ..., An)
    A = [set(bundle) for bundle in initial_A]
    
    # Visited <- {canonical(A)}
    visited = {canonical(A)}
    
    while True:
        if is_efx(A, V):
            return A  # EFX allocation found
            
        envy_pairs = get_efx_envy_pairs(A, V)
        
        if not envy_pairs:
            return A  # No EFX-envy pairs remain
            
        # We use this flag to simulate the "continue while" behavior from the pseudocode
        while_continued = False
        
        for i, j in envy_pairs:
            # g <- arg min_{h in Aj} w_{j,h} (Least-valued item in Aj by agent j)
            g = min(A[j], key=lambda h: w(j, h))
            
            # candidate <- A with g moved from Aj to Ai
            candidate = [set(bundle) for bundle in A]
            candidate[j].remove(g)
            candidate[i].add(g)
            
            PV = get_pairwise_violations(candidate)
            TV = get_triple_violations(candidate)
            
            if not PV and not TV:
                A = candidate
                if canonical(A) in visited:
                    raise RuntimeError("Cycle detected (Halt)")
                visited.add(canonical(A))
                while_continued = True
                break  # continue while
                
            best_delta = -math.inf
            best_alloc = None
            
            for p, q, a, b in PV:
                swapped = swap_a_b(candidate, p, q, a, b)
                delta = phi(swapped) - phi(candidate)
                if delta > best_delta:
                    best_delta = delta
                    best_alloc = swapped
                    
            for p, q, r, a, b, c in TV:
                swapped = cyclic_swap(candidate, p, q, r, a, b, c)
                delta = phi(swapped) - phi(candidate)
                if delta > best_delta:
                    best_delta = delta
                    best_alloc = swapped
                    
            if canonical(best_alloc) in visited:
                raise RuntimeError("Cycle detected (Halt)")
                
            A = best_alloc
            visited.add(canonical(A))
            while_continued = True
            break  # continue while
            
        # If the inner for-loop completes without hitting a 'break' (which acts as continue while), 
        # we need to safely break the outer loop to prevent an infinite loop, 
        # though algorithmically the 'break' should always be hit in this algorithm's flow.
        if not while_continued:
            break
            
    return A