from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Namespace


XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


def _local_name(uri: URIRef | str) -> str:
    """Get the local name from a URI (after last # or /)."""
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rstrip("/").rsplit("/", 1)[-1]


def _first_literal(g: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
    """Get the first string literal for (subject, predicate, ?o), or None."""
    for o in g.objects(subject, predicate):
        return str(o)
    return None


def _infer_base_iri(uris: List[str]) -> Optional[str]:
    """Heuristic: get common prefix of class/property IRIs and cut at last / or #."""
    if not uris:
        return None
    prefix = uris[0]
    for u in uris[1:]:
        # common prefix
        while not u.startswith(prefix) and prefix:
            prefix = prefix[:-1]
        if not prefix:
            break
    if not prefix:
        return None
    # cut at last / or #
    cut_positions = [prefix.rfind("/"), prefix.rfind("#")]
    cut = max(cut_positions)
    if cut <= 0:
        return prefix
    return prefix[: cut + 1]


def _map_xsd_datatype(dt: URIRef) -> str:
    """Map XSD datatype IRI to a simple type string."""
    if not str(dt).startswith(str(XSD)):
        return _local_name(dt)

    name = _local_name(dt).lower()
    if name in {"string", "normalizedstring"}:
        return "string"
    if name in {"int", "integer", "long", "short", "byte", "nonnegativeinteger"}:
        return "integer"
    if name in {"float", "double", "decimal"}:
        return "number"
    if name in {"boolean"}:
        return "boolean"
    if name in {"date", "datetime", "time"}:
        return name  # "date", "datetime", etc.
    return name


def ontology_to_llm_json_dict(input_path: str | Path) -> Dict[str, Any]:
    """
    Load an OWL/RDF ontology and return a *LLM-friendly* JSON-like dict.

    Structure:

    {
      "base_iri": "...",
      "classes": {
        "User": {
          "iri": "...Users/User",
          "label": "User",
          "description": "Optional human-readable comment",
          "properties": {
            "fullname": {
              "iri": "...Users/fullname",
              "kind": "datatype",
              "type": "string",
              "description": "Full name of the user"
            },
            "hasLocation": {
              "iri": "...Users/hasLocation",
              "kind": "object",
              "target_classes": ["Location"],
              "description": "User's location"
            }
          }
        },
        "Location": { ... }
      },
      "orphan_properties": { ... }  # properties without explicit domain
    }
    """
    input_path = str(input_path)
    g = Graph()
    g.parse(input_path)

    classes: Dict[str, Dict[str, Any]] = {}
    all_uris: List[str] = []

    # --- Collect classes ---
    for c in g.subjects(RDF.type, OWL.Class):
        c_iri = str(c)
        all_uris.append(c_iri)
        cname = _local_name(c)
        label = _first_literal(g, c, RDFS.label) or cname
        comment = _first_literal(g, c, RDFS.comment)

        if cname not in classes:
            classes[cname] = {
                "iri": c_iri,
                "label": label,
                "description": comment,
                "properties": {},  # filled later
            }

    # --- Collect properties (object + datatype) ---
    orphan_properties: Dict[str, Any] = {}

    def handle_property(prop_uri: URIRef, kind: str) -> None:
        p_iri = str(prop_uri)
        all_uris.append(p_iri)
        pname = _local_name(prop_uri)
        label = _first_literal(g, prop_uri, RDFS.label) or pname
        comment = _first_literal(g, prop_uri, RDFS.comment)

        domains = list(g.objects(prop_uri, RDFS.domain))
        ranges = list(g.objects(prop_uri, RDFS.range))

        if kind == "datatype":
            # pick first datatype range, if present
            if ranges:
                datatype = _map_xsd_datatype(ranges[0])
            else:
                datatype = "string"  # safe default
            prop_payload: Dict[str, Any] = {
                "iri": p_iri,
                "kind": "datatype",
                "type": datatype,
                "label": label,
                "description": comment,
            }
        else:  # object property
            target_classes = [_local_name(r) for r in ranges] if ranges else []
            prop_payload = {
                "iri": p_iri,
                "kind": "object",
                "target_classes": target_classes,
                "label": label,
                "description": comment,
            }

        # Attach to domain classes if we have them
        if domains:
            for d in domains:
                dname = _local_name(d)
                if dname not in classes:
                    classes[dname] = {
                        "iri": str(d),
                        "label": dname,
                        "description": _first_literal(g, d, RDFS.comment),
                        "properties": {},
                    }
                classes[dname]["properties"][pname] = prop_payload
        else:
            # No explicit domain → store under 'orphan_properties'
            orphan_properties[pname] = prop_payload

    # Object properties
    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        handle_property(p, "object")

    # Datatype properties
    for p in g.subjects(RDF.type, OWL.DatatypeProperty):
        handle_property(p, "datatype")

    base_iri = _infer_base_iri(all_uris)

    result: Dict[str, Any] = {
        "base_iri": base_iri,
        "classes": classes,
    }
    if orphan_properties:
        result["orphan_properties"] = orphan_properties

    return result


# Example usage in code (no CLI):
if __name__ == "__main__":
    onto_path = "/Users/p.salimi1/Documents/MAYA/maya/framework/ontologies/UsersData.rdf"  # <- change to your file path
    data = ontology_to_llm_json_dict(onto_path)

    # Optional: write to JSON file next to the ontology
    output_path = str(Path(onto_path).with_suffix(".llm.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote LLM-friendly JSON ontology to: {output_path}")
