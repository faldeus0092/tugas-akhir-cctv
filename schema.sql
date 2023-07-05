--
-- PostgreSQL database dump
--

-- Dumped from database version 15.3
-- Dumped by pg_dump version 15.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: cctvs; Type: TABLE; Schema: public; Owner: root
--

CREATE TABLE public.cctvs (
    id integer NOT NULL,
    cctv_number integer,
    name text
);


ALTER TABLE public.cctvs OWNER TO root;

--
-- Name: cctvs_id_seq; Type: SEQUENCE; Schema: public; Owner: root
--

CREATE SEQUENCE public.cctvs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.cctvs_id_seq OWNER TO root;

--
-- Name: cctvs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: root
--

ALTER SEQUENCE public.cctvs_id_seq OWNED BY public.cctvs.id;


--
-- Name: footages; Type: TABLE; Schema: public; Owner: root
--

CREATE TABLE public.footages (
    cctv_id integer,
    image_path text,
    num_detections integer,
    date timestamp without time zone
);


ALTER TABLE public.footages OWNER TO root;

--
-- Name: cctvs id; Type: DEFAULT; Schema: public; Owner: root
--

ALTER TABLE ONLY public.cctvs ALTER COLUMN id SET DEFAULT nextval('public.cctvs_id_seq'::regclass);


--
-- Name: cctvs cctvs_pkey; Type: CONSTRAINT; Schema: public; Owner: root
--

ALTER TABLE ONLY public.cctvs
    ADD CONSTRAINT cctvs_pkey PRIMARY KEY (id);


--
-- Name: footages footages_cctv_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: root
--

ALTER TABLE ONLY public.footages
    ADD CONSTRAINT footages_cctv_id_fkey FOREIGN KEY (cctv_id) REFERENCES public.cctvs(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

